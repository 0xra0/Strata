// Read-only SVE1 records: image rows share the encoder's file pages instead of
// allocating another anonymous copy. Keep this object alive through prefill.
#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {
class VisionRecords {
public:
    struct Image { int32_t n, nx, ny; const float* data; };
    VisionRecords() = default;
    VisionRecords(const VisionRecords&) = delete;
    VisionRecords& operator=(const VisionRecords&) = delete;
    ~VisionRecords() { close(); }
    const std::vector<Image>& images() const { return images_; }
    size_t floats() const { return floats_; }
    void close() {
        images_.clear();
        floats_ = 0;
#if defined(__linux__)
        if (data_) (void) munmap(data_, bytes_);
#endif
        data_ = nullptr;
        bytes_ = 0;
    }
    bool open(const std::string& path, int64_t width, uint64_t max_rows, std::string& err) {
        close();
        err.clear();
#if defined(__linux__)
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { err = "cannot open " + path; return false; }
        struct stat st{};
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
            (uint64_t) st.st_size > (uint64_t) std::numeric_limits<size_t>::max()) {
            ::close(fd);
            err = "bad embeddings file size";
            return false;
        }
        bytes_ = (size_t) st.st_size;
        void* p = mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) { bytes_ = 0; err = "cannot map embeddings file"; return false; }
        data_ = p;
        size_t at = 0;
        uint64_t rows = 0;
        auto fail = [&](const char* why) { err = why; close(); return false; };
        while (at < bytes_) {
            int32_t h[5];
            if (bytes_ - at < sizeof h) return fail("short embeddings header");
            std::memcpy(h, (const char*) data_ + at, sizeof h);
            at += sizeof h;
            if (h[0] != 0x31455653 || h[1] < 1 || h[2] < 1 || h[3] < 1 || h[4] < 1 || h[4] != width ||
                (int64_t) h[2] * h[3] != h[1]) return fail("bad embeddings record geometry");
            if ((uint64_t) h[1] > max_rows - rows) return fail("embeddings rows exceed the prompt's image tokens");
            const uint64_t count = (uint64_t) h[1] * (uint64_t) h[4];
            if (count > (bytes_ - at) / sizeof(float)) return fail("short embeddings file");
            // SVE1's 20-byte header and float payload keep every row 4-byte aligned.
            images_.push_back({h[1], h[2], h[3], (const float*) ((const char*) data_ + at)});
            floats_ += (size_t) count;
            rows += (uint64_t) h[1];
            at += (size_t) count * sizeof(float);
        }
        return true;
#else
        (void) path; (void) width; (void) max_rows;
        err = "mapped vision records require Linux";
        return false;
#endif
    }
private:
    void* data_ = nullptr;
    size_t bytes_ = 0, floats_ = 0;
    std::vector<Image> images_;
};
}  // namespace strata::core
