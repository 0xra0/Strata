// Read-ahead of predicted pageable RAM experts, from the routing lookahead worker.
// This never reads/writes expert values and never locks or discards their pages.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core::detail {
struct ResidentReadAheadStats {
    uint64_t probes, advised_bytes, errors, budget_skips, pressure_skips, unknown_skips, no_swap_skips, microseconds;
};

class ResidentSwapGate {
public:
    template<class ReadSwap>
    bool active(uint64_t now, ReadSwap read) {
        if (!sampled_ || now < sampled_at_ || now - sampled_at_ >= 1000000) {
            sampled_ = true; sampled_at_ = now; known_ = read(bytes_);
        }
        return !known_ || bytes_ > 0;
    }
private:
    uint64_t sampled_at_ = 0, bytes_ = 0;
    bool sampled_ = false, known_ = false;
};

class ResidentReadAhead {
public:
    static constexpr uint64_t kPeriodUs = 100000, kBudget = 128ull << 20, kFloor = 2560ull << 20;
    static constexpr size_t kMaxBlob = 8u << 20, kMaxRuns = 8;

    ResidentReadAheadStats stats() const {
        return {probes_.load(), advised_.load(), errors_.load(), budget_skips_.load(),
                pressure_skips_.load(), unknown_skips_.load(), no_swap_skips_.load(), microseconds_.load()};
    }

    // Called by one worker only. Readers of stats() may run concurrently. Callback
    // injection tests the exact production selection/pressure/budget path without swap.
    template<class Probe, class Advise, class Available>
    void hint_with(const uint8_t* ptr, size_t bytes, size_t page, uint64_t now,
                   Probe probe, Advise advise, Available available) {
        if (!ptr || !bytes || bytes > kMaxBlob || !page) return;
        const uintptr_t p = (uintptr_t) ptr, limit = (std::numeric_limits<uintptr_t>::max)();
        if (bytes > limit - p) return;
        const uintptr_t start = p - p % page, end = p + bytes;
        const size_t tail = end % page ? page - end % page : 0;
        if (tail > limit - end) return;
        const size_t length = (size_t) (end + tail - start), pages = length / page;
        bits_.resize(pages);
        ++probes_;
        if (!probe((const void*) start, length, bits_.data())) { ++errors_; return; }
        if (std::all_of(bits_.begin(), bits_.end(), [](auto b) { return (b & 1) != 0; })) return;
        if (!period_set_ || now < period_at_ || now - period_at_ >= kPeriodUs) {
            period_set_ = true; period_at_ = now; used_ = 0;
            memory_known_ = available(remaining_);
        }
        if (!memory_known_) { ++unknown_skips_; return; }
        size_t runs = 0;
        for (size_t i = 0; i < pages && runs < kMaxRuns;) {
            if (bits_[i] & 1) { ++i; continue; }
            const size_t first = i++;
            while (i < pages && !(bits_[i] & 1)) ++i;
            const size_t n = (i - first) * page;
            if (n > kBudget - used_) { ++budget_skips_; return; }
            if (remaining_ <= kFloor || n > remaining_ - kFloor) { ++pressure_skips_; return; }
            // Reserve even when the hint fails; repeated failures must stay bounded.
            used_ += n; remaining_ -= n; ++runs;
            if (advise((const void*) (start + first * page), n)) advised_ += n;
            else ++errors_;
        }
    }

    template<class Available>
    void hint(const uint8_t* ptr, size_t bytes, Available available) {
#if defined(__linux__)
        const auto t0 = std::chrono::steady_clock::now();
        const auto now = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t0.time_since_epoch()).count();
        // Healthy runs do not need a mincore syscall per expert. Sample VmSwap
        // once per second; unknown swap status retains the guarded probe path.
        if (!swap_gate_.active(now, [](uint64_t& bytes) {
            FILE* f = std::fopen("/proc/self/status", "r");
            if (!f) return false;
            char line[256]; unsigned long long kib = 0; bool found = false;
            while (std::fgets(line, sizeof line, f))
                if (std::sscanf(line, "VmSwap: %llu kB", &kib) == 1) { found = true; break; }
            std::fclose(f);
            if (!found || kib > (std::numeric_limits<uint64_t>::max)() / 1024) return false;
            bytes = (uint64_t) kib * 1024; return true;
        })) {
            ++no_swap_skips_;
            return;
        }
        static const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) return;
        hint_with(ptr, bytes, (size_t) page, now,
                  [](const void* p, size_t n, unsigned char* b) { return mincore((void*) p, n, b) == 0; },
                  [](const void* p, size_t n) { return madvise((void*) p, n, MADV_WILLNEED) == 0; }, available);
        microseconds_ += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0).count();
#else
        (void) ptr; (void) bytes; (void) available;
#endif
    }
private:
    std::vector<unsigned char> bits_;
    ResidentSwapGate swap_gate_;
    uint64_t period_at_ = 0, used_ = 0, remaining_ = 0;
    bool period_set_ = false, memory_known_ = false;
    std::atomic<uint64_t> probes_{0}, advised_{0}, errors_{0}, budget_skips_{0}, pressure_skips_{0},
                          unknown_skips_{0}, no_swap_skips_{0}, microseconds_{0};
};
} // namespace strata::core::detail
