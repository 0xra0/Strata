// Exact prompt reads without growing the model's page cache with cold experts.
#pragma once
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace strata::core::detail {
struct PromptReadStats {
    uint64_t cached = 0, dontcache = 0, direct = 0, buffered = 0;
    uint64_t unsupported = 0, errors = 0;
};
struct PromptReadCounters {
    std::atomic<uint64_t> cached{0}, dontcache{0}, direct{0}, buffered{0}, unsupported{0}, errors{0};
    PromptReadStats snapshot() const {
        return {cached.load(), dontcache.load(), direct.load(), buffered.load(), unsupported.load(), errors.load()};
    }
};
enum class PromptReadResult { complete, unsupported, failed };
struct PromptReadState { std::atomic<int> dontcache{0}; std::atomic<bool> direct{true}; };

// Callbacks must either supply the entire exact range or report failure. A
// failed attempt may write a prefix; every fallback overwrites the whole range.
template<class Cached, class NoCache, class Direct, class Buffered>
bool prompt_read_range(PromptReadState& state, PromptReadCounters& count, uint64_t bytes,
                       Cached cached, NoCache nocache, Direct direct, Buffered buffered) {
    if (cached()) { count.cached.fetch_add(bytes); return true; }
    if (state.dontcache.load() >= 0) {
        const auto result = nocache();
        if (result == PromptReadResult::complete) { count.dontcache.fetch_add(bytes); return true; }
        if (result == PromptReadResult::unsupported) {
            if (state.dontcache.exchange(-1) >= 0) count.unsupported.fetch_add(1);
        } else count.errors.fetch_add(1);
    }
    if (state.direct.load()) {
        const auto result = direct();
        if (result == PromptReadResult::complete) { count.direct.fetch_add(bytes); return true; }
        if (result == PromptReadResult::unsupported) state.direct.store(false);
        else count.errors.fetch_add(1);
    }
    if (buffered()) { count.buffered.fetch_add(bytes); return true; }
    count.errors.fetch_add(1);
    return false;
}

class PromptFileReader {
public:
    PromptFileReader() = default;
    PromptFileReader(const PromptFileReader&) = delete;
    PromptFileReader& operator=(const PromptFileReader&) = delete;
    ~PromptFileReader() { close(); }
    bool active() const { return !files_.empty(); }
    PromptReadStats stats() const { return count_.snapshot(); }
    bool open(const std::vector<std::string>& paths, std::string& why) {
        close();
#if defined(__linux__)
        for (const auto& path : paths) {
            auto file = std::make_unique<File>();
            file->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (file->fd < 0) { why = std::strerror(errno); close(); return false; }
            // This handle is separate from decode's: its advice cannot change
            // decode's read-ahead policy. RWF_DONTCACHE preserves existing cache.
            (void) posix_fadvise(file->fd, 0, 0, POSIX_FADV_RANDOM);
            file->direct_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
            if (file->direct_fd < 0) file->state.direct.store(false);
            files_.push_back(std::move(file));
        }
        if (files_.empty()) { why = "no expert files"; return false; }
        why = "cached spans stay cached; cold spans use RWF_DONTCACHE or O_DIRECT; buffered fallback counted";
        return true;
#else
        (void) paths; why = "Linux only"; return false;
#endif
    }
    void close() { files_.clear(); }
    bool read(size_t file_index, const uint8_t* mapped, uint64_t offset, uint64_t bytes, uint8_t* dst) {
#if defined(__linux__)
        if (file_index >= files_.size() || !mapped || !dst || !bytes ||
            bytes > std::numeric_limits<size_t>::max() ||
            bytes > std::numeric_limits<uintptr_t>::max() - (uintptr_t) mapped ||
            offset > (uint64_t) std::numeric_limits<int64_t>::max() ||
            bytes > (uint64_t) std::numeric_limits<int64_t>::max() - offset) return false;
        File& file = *files_[file_index];
        return prompt_read_range(file.state, count_, bytes,
            [&] {
                if (!in_cache(mapped, (size_t) bytes)) return false;
                std::memcpy(dst, mapped, (size_t) bytes);
                return true;
            },
            [&] { return no_cache_read(file.fd, dst, bytes, offset); },
            [&] { return direct_read(file.direct_fd, dst, bytes, offset); },
            [&] { return buffered_read(file.fd, dst, bytes, offset); });
#else
        (void) file_index; (void) mapped; (void) offset; (void) bytes; (void) dst; return false;
#endif
    }
private:
    struct File {
        int fd = -1, direct_fd = -1;
        PromptReadState state;
        ~File() {
#if defined(__linux__)
            if (fd >= 0) ::close(fd);
            if (direct_fd >= 0) ::close(direct_fd);
#endif
        }
    };
    std::vector<std::unique_ptr<File>> files_;
    PromptReadCounters count_;
#if defined(__linux__)
    static bool in_cache(const uint8_t* p, size_t n) {
        static const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        if (!page || page > (1ull << 20)) return false;
        const uintptr_t start = (uintptr_t) p / page * page;
        const size_t length = (size_t) ((uintptr_t) p - start) + n;
        if (length > std::numeric_limits<size_t>::max() - (page - 1)) return false;
        static thread_local std::vector<unsigned char> bits;
        bits.resize((length + page - 1) / page);
        if (mincore((void*) start, length, bits.data()) != 0) return false;
        for (unsigned char bit : bits) if (!(bit & 1)) return false;
        return true;
    }
    static bool unsupported(int error) {
        return error == ENOSYS || error == EOPNOTSUPP || error == EINVAL;
    }
    static PromptReadResult no_cache_read(int fd, uint8_t* dst, uint64_t n, uint64_t offset) {
        // Stable Linux UAPI value, including builds against pre-6.14 headers.
        constexpr int kDontCache = 0x80; // RWF_DONTCACHE, include/uapi/linux/fs.h
        uint64_t done = 0;
        while (done < n) {
            iovec vector{dst + done, (size_t) std::min<uint64_t>(n - done, 1ull << 20)};
            const ssize_t got = preadv2(fd, &vector, 1, (off_t) (offset + done), kDontCache);
            if (got < 0) {
                if (errno == EINTR) continue;
                return unsupported(errno) ? PromptReadResult::unsupported : PromptReadResult::failed;
            }
            if (!got) return PromptReadResult::failed;
            done += (uint64_t) got;
        }
        return PromptReadResult::complete;
    }
    static PromptReadResult direct_read(int fd, uint8_t* dst, uint64_t n, uint64_t offset) {
        if (fd < 0) return PromptReadResult::unsupported;
        // One reusable, bounded aligned buffer per existing prompt worker.
        // No full expert/model arena and no new threads or pinned allocations.
        struct Scratch {
            uint8_t* data = nullptr;
            Scratch() { void* p = nullptr; if (posix_memalign(&p, 4096, (1u << 20) + 8192) == 0) data = (uint8_t*) p; }
            ~Scratch() { std::free(data); }
        };
        static thread_local Scratch scratch;
        if (!scratch.data) return PromptReadResult::failed;
        for (uint64_t done = 0; done < n;) {
            const uint64_t position = offset + done, start = position / 4096 * 4096;
            const size_t skip = (size_t) (position - start);
            const size_t take = (size_t) std::min<uint64_t>(n - done, 1ull << 20);
            const size_t length = (skip + take + 4095) / 4096 * 4096;
            ssize_t got;
            do { got = ::pread(fd, scratch.data, length, (off_t) start); } while (got < 0 && errno == EINTR);
            if (got < 0) return unsupported(errno) ? PromptReadResult::unsupported : PromptReadResult::failed;
            if ((uint64_t) got < skip + take) return PromptReadResult::failed;
            std::memcpy(dst + done, scratch.data + skip, take);
            done += take;
        }
        return PromptReadResult::complete;
    }
    static bool buffered_read(int fd, uint8_t* dst, uint64_t n, uint64_t offset) {
        for (uint64_t done = 0; done < n;) {
            const ssize_t got = ::pread(fd, dst + done, (size_t) std::min<uint64_t>(n - done, 1ull << 20),
                                        (off_t) (offset + done));
            if (got < 0) { if (errno == EINTR) continue; return false; }
            if (!got) return false;
            done += (uint64_t) got;
        }
        return true;
    }
#endif
};
} // namespace strata::core::detail
