#include "strata/core/resident_readahead.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>

using strata::core::detail::ResidentReadAhead;
static int checks = 0;
static void check(bool ok, const char* why) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
int main() {
    constexpr size_t page = 4096, blob = 2u << 20;
    auto* fake = (const uint8_t*) 0x100000;
    strata::core::detail::ResidentSwapGate swap_gate;
    size_t swap_reads = 0;
    auto none = [&](uint64_t& n) { ++swap_reads; n = 0; return true; };
    check(!swap_gate.active(1, none) && !swap_gate.active(999999, none) && swap_reads == 1, "zero-swap fast path caches its sample for one second");
    check(swap_gate.active(1000001, [&](uint64_t& n) { ++swap_reads; n = 8ull << 30; return true; }), "swapped process enables page probes");
    check(swap_gate.active(1000002, none) && swap_reads == 2, "swap state does not read proc for every expert");
    check(swap_gate.active(2000001, [](uint64_t&) { return false; }), "unknown swap status retains guarded page probes");
    auto plenty = [](uint64_t& n) { n = 1ull << 40; return true; };
    auto cold = [](const void*, size_t n, unsigned char* b) { std::memset(b, 0, n / page); return true; };
    auto hot = [](const void*, size_t n, unsigned char* b) { std::memset(b, 1, n / page); return true; };
    size_t calls = 0, advised = 0;
    auto advise = [&](const void* p, size_t n) {
        check((uintptr_t) p % page == 0 && n % page == 0, "only page-aligned read-ahead ranges");
        ++calls; advised += n; return true;
    };
    ResidentReadAhead warm;
    warm.hint_with(fake, blob, page, 1, hot, advise, plenty);
    check(!calls && warm.stats().probes == 1, "already resident pages require no advice");
    ResidentReadAhead limit;
    for (int i = 0; i < 65; ++i) limit.hint_with(fake, blob, page, 1, cold, advise, plenty);
    check(calls == 64 && advised == ResidentReadAhead::kBudget, "128 MiB period limit is exact");
    check(limit.stats().budget_skips == 1, "excess read-ahead is skipped");
    limit.hint_with(fake, blob, page, 1 + ResidentReadAhead::kPeriodUs, cold, advise, plenty);
    check(calls == 65, "budget resets after 100 ms");

    ResidentReadAhead pressure;
    auto low = [](uint64_t& n) { n = ResidentReadAhead::kFloor + page - 1; return true; };
    const size_t old_calls = calls;
    pressure.hint_with(fake, page, page, 1, cold, advise, low);
    check(calls == old_calls && pressure.stats().pressure_skips == 1, "do not consume the free-memory floor");
    ResidentReadAhead unknown;
    unknown.hint_with(fake, page, page, 1, cold, advise, [](uint64_t&) { return false; });
    check(calls == old_calls && unknown.stats().unknown_skips == 1, "unknown memory declines prefetch");
    ResidentReadAhead reservation;
    auto tight = [](uint64_t& n) { n = ResidentReadAhead::kFloor + 2 * page; return true; };
    reservation.hint_with(fake, page, page, 1, cold, advise, tight);
    reservation.hint_with(fake, 2 * page, page, 2, cold, advise, tight);
    check(reservation.stats().pressure_skips == 1, "admitted hints reserve their memory within the period");

    ResidentReadAhead failed;
    size_t attempts = 0;
    for (int i = 0; i < 65; ++i)
        failed.hint_with(fake, blob, page, 1, cold, [&](const void*, size_t) { ++attempts; return false; }, plenty);
    check(attempts == 64 && failed.stats().errors == 64 && failed.stats().budget_skips == 1, "failures cannot bypass I/O limit");
    ResidentReadAhead bad_probe;
    bad_probe.hint_with(fake, page, page, 1, [](const void*, size_t, unsigned char*) { return false; }, advise, plenty);
    check(bad_probe.stats().errors == 1 && bad_probe.stats().advised_bytes == 0, "failed residency probe does not guess");
    ResidentReadAhead geometry;
    geometry.hint_with(fake + 1, page, page, 1, cold, advise, plenty);
    check(geometry.stats().advised_bytes == 2 * page, "unaligned expert boundaries cover the correct pages");
    geometry.hint_with((const uint8_t*) (UINTPTR_MAX - 2), 8, page, 1, cold, advise, plenty);
    geometry.hint_with(nullptr, page, page, 1, cold, advise, plenty);
    check(geometry.stats().probes == 1, "invalid bounds rejected before probe");
    ResidentReadAhead fragmented;
    fragmented.hint_with(fake, 32 * page, page, 1, [](const void*, size_t n, unsigned char* b) {
        for (size_t i = 0; i < n / page; ++i) b[i] = (unsigned char) (i % 2);
        return true;
    }, advise, plenty);
    check(fragmented.stats().advised_bytes == 8 * page, "fragmented cold ranges cap at eight syscalls per expert");

    // The host has no swap configured. Exercise actual mincore/MADV_WILLNEED on
    // an anonymous warm allocation and a read-only file mapping, not fake swap.
    const size_t real_page = (size_t) sysconf(_SC_PAGESIZE), real_size = 2u << 20;
    auto* anon = (uint8_t*) mmap(nullptr, real_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(anon != MAP_FAILED, "allocate anonymous expert fixture");
    for (size_t i = 0; i < real_size; ++i) anon[i] = (uint8_t) (i * 31 + (i >> 10));
    ResidentReadAhead resident;
    resident.hint(anon + 7, real_size - 14, plenty);
    check(resident.stats().advised_bytes == 0 && (resident.stats().no_swap_skips == 1 || resident.stats().probes == 1),
          "actual hot/zero-swap path issues no read-ahead");
    char name[] = "/tmp/strata-read-ahead-XXXXXX";
    const int fd = mkstemp(name); check(fd >= 0, "create immutable file fixture"); unlink(name);
    check(write(fd, anon, real_size) == (ssize_t) real_size && fsync(fd) == 0, "write exact fixture bytes");
    const auto* mapping = (const uint8_t*) mmap(nullptr, real_size, PROT_READ, MAP_SHARED, fd, 0);
    check(mapping != MAP_FAILED, "map read-only expert fixture");
    (void) madvise((void*) mapping, real_size, MADV_DONTNEED); // tests only: read-only file pages, never anonymous weights
    (void) posix_fadvise(fd, 0, (off_t) real_size, POSIX_FADV_DONTNEED);
    std::vector<unsigned char> bits(real_size / real_page);
    check(mincore((void*) mapping, real_size, bits.data()) == 0, "observe real file residency");
    const size_t cold_pages = (size_t) std::count_if(bits.begin(), bits.end(), [](auto b) { return !(b & 1); });
    ResidentReadAhead actual;
    actual.hint_with(mapping, real_size, real_page, 1,
        [](const void* p, size_t n, unsigned char* b) { return mincore((void*) p, n, b) == 0; },
        [](const void* p, size_t n) { return madvise((void*) p, n, MADV_WILLNEED) == 0; }, plenty);
    check(actual.stats().errors == 0, "actual read-ahead system call succeeds");
    if (cold_pages) check(actual.stats().advised_bytes > 0, "cold file pages exercise the production hint path");
    check(std::memcmp(mapping, anon, real_size) == 0, "read-ahead preserves every expert byte");
    for (size_t i = 0; i < real_size; ++i)
        if (anon[i] != (uint8_t) (i * 31 + (i >> 10))) { check(false, "anonymous weights changed"); }
    std::printf("resident read-ahead: %d checks passed; real file cold pages=%zu, %.1f MiB hints; exact bytes retained; host swap unavailable\n",
                checks, cold_pages, (double) actual.stats().advised_bytes / (1 << 20));
    munmap((void*) mapping, real_size); munmap(anon, real_size); close(fd);
}
