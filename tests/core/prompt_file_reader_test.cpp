#include "strata/core/prompt_file_reader.hpp"
#include <cstdio>
#include <thread>
#include <array>
#include <stdexcept>
using namespace strata::core::detail;
static int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) throw std::runtime_error(#x); } while (0)

static void policy() {
    for (bool hot : {false, true})
    for (auto no_cache : {PromptReadResult::complete, PromptReadResult::unsupported, PromptReadResult::failed})
    for (auto direct : {PromptReadResult::complete, PromptReadResult::unsupported, PromptReadResult::failed})
    for (bool buffered : {false, true}) {
        PromptReadState state; PromptReadCounters count;
        std::array<int, 4> calls{};
        const bool got = prompt_read_range(state, count, 73,
            [&] { ++calls[0]; return hot; }, [&] { ++calls[1]; return no_cache; },
            [&] { ++calls[2]; return direct; }, [&] { ++calls[3]; return buffered; });
        const bool first = !hot, second = first && no_cache != PromptReadResult::complete;
        const bool third = second && direct != PromptReadResult::complete;
        CHECK(got == (hot || no_cache == PromptReadResult::complete || direct == PromptReadResult::complete || buffered));
        CHECK((calls == std::array<int, 4>{1, first, second, third}));
        const auto stats = count.snapshot();
        CHECK(stats.cached == (hot ? 73u : 0u));
        CHECK(stats.dontcache == (first && no_cache == PromptReadResult::complete ? 73u : 0u));
        CHECK(stats.direct == (second && direct == PromptReadResult::complete ? 73u : 0u));
        CHECK(stats.buffered == (third && buffered ? 73u : 0u));
    }
    // Unsupported modes stop being retried; buffered reads overwrite any prefix
    // left by a failed no-cache/direct attempt. No stale weight bytes escape.
    PromptReadState state; PromptReadCounters count;
    std::array<uint8_t, 23> bytes{};
    int calls = 0;
    for (int i = 0; i < 12; ++i) {
        CHECK(prompt_read_range(state, count, bytes.size(), [] { return false; },
            [&] { ++calls; bytes.fill(8); return PromptReadResult::unsupported; },
            [&] { ++calls; bytes.fill(9); return PromptReadResult::unsupported; },
            [&] { bytes.fill(42); return true; }));
        CHECK(std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == 42; }));
    }
    CHECK(calls == 2); CHECK(count.snapshot().unsupported == 1);
}

#if defined(__linux__)
struct Fixture {
    char path[64] = "/tmp/strata-prompt-reader-XXXXXX";
    int fd = -1; uint8_t* map = nullptr;
    std::vector<uint8_t> data;
    explicit Fixture(size_t n) : data(n) {
        fd = mkstemp(path); if (fd < 0) throw std::runtime_error("mkstemp");
        for (size_t i = 0; i < n; ++i) data[i] = (uint8_t) ((i * 137 + i / 8191) % 251);
        for (size_t done = 0; done < n;) {
            const ssize_t got = ::write(fd, data.data() + done, n - done);
            if (got <= 0) throw std::runtime_error("write");
            done += (size_t) got;
        }
        if (fsync(fd)) throw std::runtime_error("fsync");
        map = (uint8_t*) mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) throw std::runtime_error("mmap");
    }
    size_t cached() const {
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        std::vector<unsigned char> bits((data.size() + page - 1) / page);
        if (mincore(map, data.size(), bits.data())) throw std::runtime_error("mincore");
        size_t count = 0; for (auto bit : bits) count += (bit & 1); return count * page;
    }
    void cold() {
        if (madvise(map, data.size(), MADV_DONTNEED) || posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED))
            throw std::runtime_error("fixture cache reset");
    }
    ~Fixture() { if (map && map != MAP_FAILED) munmap(map, data.size()); if (fd >= 0) ::close(fd); unlink(path); }
};

static void files() {
    Fixture first((8u << 20) + 137), second((5u << 20) + 73);
    PromptFileReader reader; std::string why;
    CHECK(reader.open({first.path, second.path}, why)); CHECK(reader.active());
    // Hot spans never call a disk read, and continue returning the exact bytes.
    std::vector<uint8_t> result(first.data.size());
    CHECK(std::memcmp(first.map, first.data.data(), first.data.size()) == 0);
    CHECK(reader.read(0, first.map + 17, 17, first.data.size() - 17, result.data()));
    CHECK(std::memcmp(result.data(), first.data.data() + 17, first.data.size() - 17) == 0);
    CHECK(reader.stats().cached == first.data.size() - 17);
    first.cold(); second.cold();
    const size_t before = first.cached() + second.cached();
    // Three role slices, two shards, unaligned boundaries and the final EOF page.
    const std::array<size_t, 3> starts{17, 713, (2u << 20) + 39};
    const std::array<size_t, 3> lengths{(3u << 20) + 13, (2u << 20) + 91, second.data.size() - starts[2]};
    result.resize(lengths[0] + lengths[1] + lengths[2]);
    size_t at = 0;
    for (size_t r = 0; r < 3; ++r) {
        Fixture& file = r == 0 ? first : second;
        CHECK(reader.read(r == 0 ? 0 : 1, file.map + starts[r], starts[r], lengths[r], result.data() + at));
        CHECK(std::memcmp(result.data() + at, file.data.data() + starts[r], lengths[r]) == 0);
        at += lengths[r];
    }
    const size_t after = first.cached() + second.cached();
    const auto cold = reader.stats();
    std::printf("real cold slices: cache before=%zu after=%zu; RWF_DONTCACHE=%llu O_DIRECT=%llu fallback=%llu bytes\n",
        before, after, (unsigned long long) cold.dontcache, (unsigned long long) cold.direct, (unsigned long long) cold.buffered);
    if (before == 0 && (cold.dontcache + cold.direct) && !cold.buffered) CHECK(after < (1u << 20));
    first.cold(); second.cold();
    at = 0;
    for (size_t r = 0; r < 3; ++r) {
        Fixture& file = r == 0 ? first : second;
        std::memcpy(result.data() + at, file.map + starts[r], lengths[r]);
        CHECK(std::memcmp(result.data() + at, file.data.data() + starts[r], lengths[r]) == 0);
        at += lengths[r];
    }
    std::printf("same slices with ordinary mapped copies: cache=%zu bytes\n", first.cached() + second.cached());
    first.cold(); second.cold();
    // Concurrent existing prompt workers share descriptor capability state only.
    std::atomic<int> failures{0}; std::vector<std::thread> workers;
    for (size_t thread = 0; thread < 8; ++thread) workers.emplace_back([&, thread] {
        std::vector<uint8_t> out((1u << 20) + 11);
        for (size_t round = 0; round < 16; ++round) {
            const size_t start = (thread * 131071 + round * 7919) % (second.data.size() - out.size());
            if (!reader.read(1, second.map + start, start, out.size(), out.data()) ||
                std::memcmp(out.data(), second.data.data() + start, out.size())) ++failures;
        }
    });
    for (auto& worker : workers) worker.join();
    CHECK(failures.load() == 0);
    CHECK(!reader.read(2, first.map, 0, 1, result.data()));
    CHECK(!reader.read(0, first.map, 0, 0, result.data()));
    CHECK(!reader.read(0, nullptr, 0, 1, result.data()));
    CHECK(!reader.read(0, first.map, UINT64_MAX, 1, result.data()));
    CHECK(!reader.read(0, first.map, 0, 1, nullptr));
    reader.close(); CHECK(!reader.active());
    CHECK(!reader.read(0, first.map, 0, 1, result.data()));
    CHECK(!reader.open({first.path, "/nonexistent/strata-experts"}, why)); CHECK(!reader.active());
}
#endif
int main() {
    try { policy();
#if defined(__linux__)
        files();
#endif
        std::printf("prompt_file_reader_test: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
