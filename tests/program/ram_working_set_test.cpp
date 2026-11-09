#include "strata/program/ram_working_set.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <unistd.h>

using strata::program::ram_working_set_plan;
static int checks = 0;
static void check(bool ok, const char* why) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
int main() {
    constexpr size_t layers = 4, experts = 16, count = layers * experts, blob = 64 * 1024, ram_per_layer = 6;
    std::array<uint64_t, layers> sizes; sizes.fill(blob);
    std::array<float, count> heat; heat.fill(1.0f);
    std::array<int32_t, count> gpu; gpu.fill(-1);
    std::array<uint8_t, count> resident{};
    for (size_t l = 0; l < layers; ++l) {
        for (size_t e = 0; e < 4; ++e) gpu[l * experts + e] = (int32_t) (l * 4 + e);
        for (size_t e = 4; e < 10; ++e) resident[l * experts + e] = 1;
        heat[l * experts + 10] = 8;
    }
    const auto unchanged_gpu = gpu;
    auto plan = ram_working_set_plan(heat.data(), gpu.data(), resident.data(), sizes.data(), layers, experts);
    check(plan.size() == layers, "colder active RAM entries can be replaced (old unused-only rule cannot)");
    for (auto& p : plan) check(p.in == 10 && p.out == 4 && p.bytes == blob, "deterministic same-layer owners");
    check(ram_working_set_plan(heat.data(), gpu.data(), resident.data(), sizes.data(), layers, experts, 4, blob - 1).empty(), "never exceed byte budget");
    check(ram_working_set_plan(heat.data(), gpu.data(), resident.data(), sizes.data(), layers, experts, 1).size() == 1, "move cap enforced");
    heat.fill(2.0f);
    check(ram_working_set_plan(heat.data(), gpu.data(), resident.data(), sizes.data(), layers, experts).empty(), "equal hotness must not churn");
    heat[10] = std::numeric_limits<float>::infinity();
    heat[11] = std::numeric_limits<float>::quiet_NaN();
    check(ram_working_set_plan(heat.data(), gpu.data(), resident.data(), sizes.data(), layers, experts).empty(), "ignore invalid routing counts");

    // Real immutable file bytes and a fixed host allocation. Trace replay measures
    // logical expert reads, not target SSD traffic or full-model tokens/sec.
    char name[] = "/tmp/strata-ram-working-set-XXXXXX";
    int fd = mkstemp(name); check(fd >= 0, "create expert file fixture"); unlink(name);
    std::vector<uint8_t> bounce(blob), original(count * blob);
    for (size_t e = 0; e < count; ++e)
        for (size_t b = 0; b < blob; ++b) original[e * blob + b] = (uint8_t) (e * 37 + b * 11 + (b >> 8));
    check(write(fd, original.data(), original.size()) == (ssize_t) original.size(), "write complete fixture");
    std::vector<uint8_t> ram(layers * ram_per_layer * blob);
    const auto* allocation = ram.data();
    std::array<int32_t, count> slots; slots.fill(-1);
    for (size_t l = 0; l < layers; ++l) for (size_t e = 4; e < 10; ++e) {
        const size_t i = l * experts + e, slot = l * ram_per_layer + e - 4;
        slots[i] = (int32_t) slot;
        std::memcpy(ram.data() + slot * blob, original.data() + i * blob, blob);
    }
    auto read_expert = [&](size_t e, uint8_t* dst) {
        check(pread(fd, dst, blob, (off_t) (e * blob)) == (ssize_t) blob, "read exact immutable expert bytes");
    };
    size_t baseline_reads = 0, routed_reads = 0, promotion_reads = 0;
    heat.fill(1.0f);
    for (int round = 0; round < 512; ++round) {
        for (auto& x : heat) x *= 0.97f;
        // Halfway through, the hot file-backed set changes, as a conversation can.
        for (size_t l = 0; l < layers; ++l) {
            for (size_t j = 0; j < 4; ++j) {
                const size_t e = l * experts + (round < 256 ? 10 : 4) + j;
                if (round < 256) ++baseline_reads; // the original static RAM set has 4..9
                heat[e] += 1.0f;
                const uint8_t* p;
                if (slots[e] >= 0) p = ram.data() + (size_t) slots[e] * blob;
                else { read_expert(e, bounce.data()); p = bounce.data(); ++routed_reads; }
                check(std::memcmp(p, original.data() + e * blob, blob) == 0, "every routed expert retains exact weights");
            }
        }
        if (round % 4 == 0) {
            for (size_t i = 0; i < count; ++i) resident[i] = slots[i] >= 0;
            plan = ram_working_set_plan(heat.data(), gpu.data(), resident.data(), sizes.data(), layers, experts);
            for (auto& p : plan) {
                const size_t in = (size_t) p.layer * experts + (size_t) p.in;
                const size_t out = (size_t) p.layer * experts + (size_t) p.out;
                check(gpu[in] < 0 && gpu[out] < 0 && slots[in] < 0 && slots[out] >= 0, "never mutate GPU or transient ownership");
                read_expert(in, bounce.data()); ++promotion_reads;
                std::memcpy(ram.data() + (size_t) slots[out] * blob, bounce.data(), blob);
                slots[in] = slots[out]; slots[out] = -1;
            }
            check(std::count_if(slots.begin(), slots.end(), [](auto s) { return s >= 0; }) == layers * ram_per_layer, "RAM capacity is constant");
            check(ram.data() == allocation && gpu == unchanged_gpu, "RAM address and GPU table never change");
            for (size_t e = 0; e < count; ++e) if (slots[e] >= 0)
                check(std::memcmp(ram.data() + (size_t) slots[e] * blob, original.data() + e * blob, blob) == 0, "all RAM weights exact after replacements");
        }
    }
    check(routed_reads + promotion_reads < baseline_reads / 4, "working set avoids repeated file reads in shifted-hotset fixture");
    std::printf("RAM working set: %d checks passed; static=%zu routed file reads, adaptive=%zu + %zu promotions; %.1f%% fewer logical reads; fixed %zu-byte RAM arena\n",
                checks, baseline_reads, routed_reads, promotion_reads,
                100.0 * (1.0 - (double) (routed_reads + promotion_reads) / baseline_reads), ram.size());
    close(fd);
}
