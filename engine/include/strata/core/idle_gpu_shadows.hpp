// Reclaim only RAM copies that have an authoritative, unchanged copy in VRAM.
// The caller holds the engine idle until restore completes. CPU-only experts
// are never included. No backing file, quantization or inference change.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#if defined(__linux__)
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core::detail {
struct GpuShadowBlob {
    size_t expert = 0, offset = 0, bytes = 0;
    int32_t slot = -1;
};
struct GpuShadowRange { size_t offset, bytes; };

// The serve loop's live table is authoritative. ExpertCache::slot_of() is the
// original profile and is NOT updated by adaptive swaps. Never infer ownership
// from that startup table when releasing or restoring host weights.
inline bool gpu_shadow_plan(const std::vector<uint64_t>& offsets,
                            const std::vector<uint64_t>& layer_bytes, size_t experts_per_layer,
                            const std::vector<int32_t>& live, size_t slots,
                            std::vector<GpuShadowBlob>& blobs, std::string& error) {
    blobs.clear();
    error.clear();
    if (!experts_per_layer || layer_bytes.size() > (std::numeric_limits<size_t>::max)() / experts_per_layer ||
        offsets.size() != layer_bytes.size() * experts_per_layer || live.size() != offsets.size()) {
        error = "idle GPU shadows: invalid live ownership table shape";
        return false;
    }
    std::vector<uint8_t> owned(slots, 0);
    for (size_t i = 0; i < live.size(); ++i) {
        const int32_t slot = live[i];
        if (slot < 0) continue;
        if ((size_t) slot >= slots || owned[(size_t) slot]++) {
            error = "idle GPU shadows: invalid or multiply owned live slot";
            blobs.clear();
            return false;
        }
        const uint64_t offset = offsets[i], bytes = layer_bytes[i / experts_per_layer];
        if (offset == (std::numeric_limits<uint64_t>::max)()) continue;
        if (!bytes || offset > (std::numeric_limits<size_t>::max)() ||
            bytes > (std::numeric_limits<size_t>::max)()) {
            error = "idle GPU shadows: invalid host expert bounds";
            blobs.clear();
            return false;
        }
        blobs.push_back({i, (size_t) offset, (size_t) bytes, slot});
    }
    return true;
}

inline bool gpu_shadow_owner_matches(const GpuShadowBlob& blob, const std::vector<int32_t>& live) {
    return blob.expert < live.size() && live[blob.expert] == blob.slot;
}

inline bool gpu_shadow_ranges(uintptr_t base, size_t total, size_t page,
                              std::vector<GpuShadowBlob>& blobs,
                              std::vector<GpuShadowRange>& ranges, std::string& error) {
    error.clear();
    ranges.clear();
    if (!base || !page || total > (std::numeric_limits<uintptr_t>::max)() - base) {
        error = "idle GPU shadows: invalid allocation bounds"; return false;
    }
    std::sort(blobs.begin(), blobs.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
    for (size_t i = 0; i < blobs.size(); ++i) {
        const auto& b = blobs[i];
        if (!b.bytes || b.slot < 0 || b.offset > total || b.bytes > total - b.offset ||
            (i && blobs[i-1].offset + blobs[i-1].bytes > b.offset)) {
            error = "idle GPU shadows: overlapping or invalid expert ranges"; return false;
        }
    }
    // Join neighbouring shadow blobs before aligning. A page shared with a CPU-only
    // blob is excluded; a page shared only by two VRAM shadows can be reclaimed.
    for (size_t i = 0; i < blobs.size();) {
        const size_t start = blobs[i].offset;
        size_t end = start + blobs[i++].bytes;
        while (i < blobs.size() && blobs[i].offset == end) { end += blobs[i].bytes; ++i; }
        const size_t misalignment = (base + start) % page;
        const size_t skip = misalignment ? page - misalignment : 0;
        if (skip >= end - start) continue;
        const size_t aligned = start + skip;
        const size_t tail = (base + end) % page;
        if (tail <= end - aligned && end - tail > aligned)
            ranges.push_back({aligned, end - tail - aligned});
    }
    return true;
}

class IdleGpuShadows {
public:
    bool active() const { return !blobs_.empty(); }
    size_t released_bytes() const { return released_; }
    size_t blobs() const { return blobs_.size(); }
    bool missing(size_t expert) const {
        return std::binary_search(missing_.begin(), missing_.end(), expert);
    }
    void clear() { blobs_.clear(); missing_.clear(); released_ = 0; arena_ = nullptr; }

    bool release(void* arena, size_t total, std::vector<GpuShadowBlob> blobs, std::string& error) {
        error.clear();
        if (active()) { error = "idle GPU shadows: restore the previous lease first"; return false; }
#if defined(__linux__)
        const long ps = sysconf(_SC_PAGESIZE);
        std::vector<GpuShadowRange> ranges;
        if (ps <= 0 || !gpu_shadow_ranges((uintptr_t) arena, total, (size_t) ps, blobs, ranges, error)) return false;
        // Every allocation and validation happens before discarding a single byte.
        std::vector<size_t> missing;
        for (const auto& b : blobs) missing.push_back(b.expert);
        std::sort(missing.begin(), missing.end());
        if (std::adjacent_find(missing.begin(), missing.end()) != missing.end()) {
            error = "idle GPU shadows: duplicate expert identity"; return false;
        }
        blobs_ = std::move(blobs);
        missing_ = std::move(missing);
        arena_ = arena;
        released_ = 0;
        for (const auto& r : ranges) {
            if (madvise((uint8_t*) arena + r.offset, r.bytes, MADV_DONTNEED) == 0) released_ += r.bytes;
            else if (error.empty()) error = std::string("idle GPU shadows: madvise: ") + std::strerror(errno);
        }
        if (!released_) clear();
        return true;
#else
        (void) arena; (void) total; (void) blobs;
        error = "idle GPU shadows: supported on Linux only";
        return false;
#endif
    }

    // A failed restore keeps the lease marked missing. Retry is safe; inference
    // must never proceed after failure. Copy exact complete blobs, including the
    // few boundary bytes retained to protect adjacent CPU-only experts.
    template<class Copy>
    bool restore(void* arena, Copy copy, std::string& error) {
        error.clear();
        if (active() && arena != arena_) { error = "idle GPU shadows: allocation changed during lease"; return false; }
        for (const auto& b : blobs_)
            if (!copy(b, (uint8_t*) arena + b.offset, error)) return false;
        clear();
        return true;
    }
private:
    std::vector<GpuShadowBlob> blobs_;
    std::vector<size_t> missing_;
    size_t released_ = 0;
    void* arena_ = nullptr;
};
} // namespace strata::core::detail
