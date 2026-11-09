// Bounded host-tier replacements. Routing and GPU ownership are inputs only.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {
struct RamReplacement { int32_t layer, in, out; float gain; uint64_t bytes; };

inline std::vector<RamReplacement> ram_working_set_plan(
        const float* heat, const int32_t* gpu, const uint8_t* resident,
        const uint64_t* layer_bytes, size_t layers, size_t experts,
        size_t max_moves = 4, uint64_t max_bytes = 8ull << 20) {
    std::vector<RamReplacement> candidates, result;
    if (!heat || !gpu || !resident || !layer_bytes || !experts || !max_moves || !max_bytes) return result;
    for (size_t l = 0; l < layers; ++l) {
        if (!layer_bytes[l] || layer_bytes[l] > max_bytes) continue;
        const size_t base = l * experts;
        int32_t incoming = -1, outgoing = -1;
        for (size_t e = 0; e < experts; ++e) {
            const size_t i = base + e;
            if (gpu[i] >= 0 || !std::isfinite(heat[i]) || heat[i] < 0) continue;
            if (resident[i]) {
                if (outgoing < 0 || heat[i] < heat[base + (size_t) outgoing]) outgoing = (int32_t) e;
            } else if (heat[i] >= 2.0f && (incoming < 0 || heat[i] > heat[base + (size_t) incoming])) {
                incoming = (int32_t) e;
            }
        }
        if (incoming < 0 || outgoing < 0) continue;
        const float hot = heat[base + (size_t) incoming], cold = heat[base + (size_t) outgoing];
        // Hysteresis: a tiny difference must not churn the same weights through SSD.
        // Unlike the old <0.5 victim rule, a colder but still active slot can move.
        if (hot <= 1.5f * cold + 0.5f) continue;
        candidates.push_back({(int32_t) l, incoming, outgoing, hot - cold, layer_bytes[l]});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.gain > b.gain; });
    uint64_t used = 0;
    for (const auto& c : candidates) {
        if (c.bytes > max_bytes - used) continue;
        result.push_back(c);
        used += c.bytes;
        if (result.size() == max_moves) break;
    }
    return result;
}
} // namespace strata::program
