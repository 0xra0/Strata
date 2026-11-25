#include "strata/core/idle_gpu_shadows.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/mman.h>

using namespace strata::core::detail;
static int checks = 0;
static void check(bool ok, const char* why) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}

int main() {
    constexpr size_t layers = 3, per_layer = 8, per_gpu = 3, count = layers * per_layer;
    constexpr uint64_t absent = ~uint64_t{0};
    const std::vector<uint64_t> bytes{4096, 8192, 12288};
    std::vector<uint64_t> offsets(count, absent);
    std::vector<int32_t> live(count, -1);
    std::vector<std::vector<uint8_t>> exact(count), gpu(layers * per_gpu);
    size_t total = 0;
    for (size_t l = 0; l < layers; ++l) {
        for (size_t e = 0; e < per_layer; ++e) {
            const size_t i = l * per_layer + e;
            exact[i].resize(bytes[l]);
            for (size_t p = 0; p < bytes[l]; ++p)
                exact[i][p] = (uint8_t) (i * 31 + p * 13 + (p >> 8) * 7);
            if (e < per_gpu) {
                live[i] = (int32_t) (l * per_gpu + e);
                gpu[(size_t) live[i]] = exact[i];
            }
            // One duplicate lend-region expert; all other RAM experts are CPU-only.
            if (e >= per_gpu - 1) { offsets[i] = total; total += bytes[l]; }
        }
    }
    const auto startup = live;   // ExpertCache::slot_of() keeps this table after swaps.
    auto* host = (uint8_t*) mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(host != MAP_FAILED, "allocate real resident expert pages");
    for (size_t i = 0; i < count; ++i)
        if (offsets[i] != absent) std::memcpy(host + offsets[i], exact[i].data(), exact[i].size());

    auto swap = [&](size_t incoming, size_t outgoing) {
        const int32_t slot = live[outgoing];
        check(slot >= 0 && live[incoming] < 0 && offsets[incoming] != absent, "adaptive swap has valid owners");
        const auto in_bytes = std::vector<uint8_t>(host + offsets[incoming], host + offsets[incoming] + exact[incoming].size());
        if (offsets[outgoing] == absent) {   // resident_stage_swaps + commit_exchanges
            std::memcpy(host + offsets[incoming], gpu[(size_t) slot].data(), exact[outgoing].size());
            offsets[outgoing] = offsets[incoming];
            offsets[incoming] = absent;
        }
        gpu[(size_t) slot] = in_bytes;
        live[outgoing] = -1;
        live[incoming] = slot;
    };
    swap(3, 0);
    std::string error;
    std::vector<GpuShadowBlob> wrong, plan;
    check(gpu_shadow_plan(offsets, bytes, per_layer, startup, gpu.size(), wrong, error), "reproduce old startup-table plan");
    bool corrupt = false;
    for (const auto& b : wrong) {
        if (gpu[(size_t) b.slot] != exact[b.expert]) corrupt = true;
    }
    check(corrupt, "old table would restore a different expert into RAM after the first swap");
    IdleGpuShadows old_shadows;
    check(old_shadows.release(host, total, wrong, error) && old_shadows.restore(host, [&](const auto& b, auto* dst, auto&) {
        std::memcpy(dst, gpu[(size_t) b.slot].data(), b.bytes);
        return true;
    }, error), "exercise the old reclaim and restore with the stale startup table");
    check(std::memcmp(host + offsets[0], exact[0].data(), exact[0].size()) != 0,
          "old path physically overwrites CPU-only expert 0 with expert 3");
    for (size_t i = 0; i < count; ++i)
        if (offsets[i] != absent) std::memcpy(host + offsets[i], exact[i].data(), exact[i].size());

    std::mt19937 random(1729);
    IdleGpuShadows shadows;
    for (int round = 0; round < 256; ++round) {
        for (size_t l = 0; l < layers; ++l) {
            std::vector<size_t> cpu, cached;
            for (size_t e = 0; e < per_layer; ++e) {
                const size_t i = l * per_layer + e;
                (live[i] < 0 ? cpu : cached).push_back(i);
            }
            swap(cpu[random() % cpu.size()], cached[random() % cached.size()]);
        }
        check(gpu_shadow_plan(offsets, bytes, per_layer, live, gpu.size(), plan, error), "plan from actual adaptive owners");
        for (const auto& b : plan)
            check(gpu_shadow_owner_matches(b, live) && gpu[(size_t) b.slot] == exact[b.expert], "every duplicate has its exact GPU weights");
        check(shadows.release(host, total, plan, error) && shadows.released_bytes() > 0, "discard only real duplicate pages");
        for (size_t i = 0; i < count; ++i) {
            if (live[i] < 0) {
                check(offsets[i] != absent && !shadows.missing(i), "CPU-only expert cannot be reclaimed");
                check(std::memcmp(host + offsets[i], exact[i].data(), exact[i].size()) == 0, "CPU-only weights intact during idle");
            }
        }
        // Same slot, wrong identity: reject before copying. Keep the lease missing.
        if (round == 0) {
            const auto b = plan.front();
            auto changed = live;
            changed[b.expert] = -1;
            check(!gpu_shadow_owner_matches(b, changed), "restore detects an expert evicted during idle");
        }
        check(shadows.restore(host, [&](const auto& b, auto* dst, auto& why) {
            if (!gpu_shadow_owner_matches(b, live)) { why = "ownership changed"; return false; }
            std::memcpy(dst, gpu[(size_t) b.slot].data(), b.bytes);
            return true;
        }, error), "restore exact bytes using live slots");
        for (size_t i = 0; i < count; ++i)
            if (offsets[i] != absent)
                check(std::memcmp(host + offsets[i], exact[i].data(), exact[i].size()) == 0, "all RAM experts exact after repeated adaptive exchanges");
    }
    auto invalid = live;
    invalid.pop_back();
    check(!gpu_shadow_plan(offsets, bytes, per_layer, invalid, gpu.size(), plan, error), "reject wrong live table size");
    invalid = live;
    invalid[0] = (int32_t) gpu.size();
    check(!gpu_shadow_plan(offsets, bytes, per_layer, invalid, gpu.size(), plan, error), "reject out-of-range live slot");
    invalid = live;
    const auto owner = (size_t) (std::find_if(live.begin(), live.end(), [](auto s) { return s >= 0; }) - live.begin());
    invalid[(owner + 1) % count] = live[owner];
    check(!gpu_shadow_plan(offsets, bytes, per_layer, invalid, gpu.size(), plan, error), "reject duplicate live ownership");
    std::printf("idle GPU ownership: %d checks passed; reproduced old wrong-expert restore; 769 adaptive swaps, 256 exact reclaim/restore rounds\n", checks);
    munmap(host, total);
}
