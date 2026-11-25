#include "strata/core/idle_gpu_shadows.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <unistd.h>

using namespace strata::core::detail;
static int checks = 0;
static void check(bool ok, const char* why) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
static size_t rss_kib() {
    size_t pages = 0, resident = 0;
    std::ifstream("/proc/self/statm") >> pages >> resident;
    return resident * (size_t) sysconf(_SC_PAGESIZE) / 1024;
}
static unsigned char pattern(size_t i) { return (unsigned char) ((i * 131 + (i >> 12) * 17) & 255); }
int main() {
    std::string error;
    std::vector<GpuShadowRange> ranges;
    std::vector<GpuShadowBlob> plan{{3, 4096, 4096, 3}, {2, 0, 4096, 2}};
    check(gpu_shadow_ranges(0x100000, 12288, 4096, plan, ranges, error), "valid neighbour plan");
    check(ranges.size() == 1 && ranges[0].offset == 0 && ranges[0].bytes == 8192, "merge only adjacent shadows");
    plan = {{1, 11, 8192, 1}, {2, 12288, 4096, 2}};
    check(gpu_shadow_ranges(0x100010, 24576, 4096, plan, ranges, error), "malloc-like unaligned base");
    for (const auto& r : ranges) {
        check((0x100010 + r.offset) % 4096 == 0, "absolute page alignment");
        check(r.offset >= 11 && r.offset + r.bytes <= 16384, "never cross shadow boundaries");
    }
    plan = {{1, 0, 4096, 1}, {2, 2048, 4096, 2}};
    check(!gpu_shadow_ranges(0x100000, 8192, 4096, plan, ranges, error), "reject overlapping blobs before reclaim");
    plan = {{1, 8192, 4096, 1}};
    check(!gpu_shadow_ranges(0x100000, 8192, 4096, plan, ranges, error), "reject outside allocation");

    // A 192 MiB host arena, half CPU-only experts and half duplicated in simulated
    // VRAM. The simulated VRAM is another anonymous mapping; this test does not
    // claim to measure PCIe restore speed on an actual GPU.
    constexpr size_t MiB = 1u << 20, total = 192 * MiB, half = total / 2;
    auto* raw = (unsigned char*) mmap(nullptr, total + 4096, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    auto* authoritative = (unsigned char*) mmap(nullptr, half, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(raw != MAP_FAILED && authoritative != MAP_FAILED, "allocate real anonymous mappings");
    auto* host = raw + 16;
    for (size_t i = 0; i < total; ++i) host[i] = pattern(i);
    std::memcpy(authoritative, host + half, half);
    const size_t before = rss_kib();
    IdleGpuShadows shadows;
    plan = {{7, half, half / 2, 1}, {9, half + half / 2, half / 2, 2}};
    check(shadows.release(host, total, plan, error), "reclaim GPU duplicates");
    const size_t idle = rss_kib();
    check(shadows.active() && shadows.released_bytes() >= half - 8192, "reclaim nearly all duplicate pages");
    check(shadows.missing(7) && shadows.missing(9) && !shadows.missing(1), "missing identity set");
    bool same = true;
    for (size_t i = 0; i < half; ++i) same &= host[i] == pattern(i);
    check(same, "every CPU-only expert byte retained");
    check(host[0] == pattern(0) && host[total - 1] == pattern(total - 1), "unaligned boundary bytes protected");
    check(!shadows.release(host, total, plan, error), "cannot discard twice before restore");
    check(!shadows.restore(host + 1, [](auto&, auto*, auto&) { return true; }, error), "allocation identity checked");
    bool first = true;
    check(!shadows.restore(host, [&](const auto& b, auto* dst, auto& e) {
        if (!first) { e = "simulated device copy failure"; return false; }
        first = false;
        std::memcpy(dst, authoritative + b.offset - half, b.bytes);
        return true;
    }, error), "failed second device copy does not publish a full restore");
    check(shadows.active() && shadows.missing(7), "failed restore remains guarded");
    check(shadows.restore(host, [&](const auto& b, auto* dst, auto&) {
        std::memcpy(dst, authoritative + b.offset - half, b.bytes); return true;
    }, error), "retry restores all exact bytes");
    same = true;
    for (size_t i = 0; i < total; ++i) same &= host[i] == pattern(i);
    check(same && !shadows.active() && error.empty(), "whole arena bit-identical after restore");
    check(shadows.restore(host, [](auto&, auto*, auto&) { return false; }, error), "empty restore is a no-op");
    plan = {{7, half, 4096, 1}, {7, half + 4096, 4096, 2}};
    check(!shadows.release(host, total, plan, error), "duplicate identities rejected before data is touched");
    plan = {{7, half, 1, 1}};
    check(shadows.release(host, total, plan, error) && !shadows.active(), "subpage expert with no safe page stays resident");
    std::printf("idle shadows: %d checks passed; fixture RSS before=%zu KiB idle=%zu KiB reduction=%.1f MiB\n",
                checks, before, idle, (double) (before - idle) / 1024);
    munmap(raw, total + 4096);
    munmap(authoritative, half);
}
