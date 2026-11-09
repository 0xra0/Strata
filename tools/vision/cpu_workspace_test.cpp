// White-box integration test: compile the actual clip.cpp implementation here so
// the private context can be tested without downloading a multimodal model. The
// static mtmd library supplies the other objects; its clip.o is not pulled in.
#include "../../third_party/llama.cpp/tools/mtmd/clip.cpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#if defined(__linux__) && defined(__GLIBC__)
#include <malloc.h>
#endif
#if defined(__linux__)
#include <unistd.h>
#endif

static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
static size_t rss_kib() {
#if defined(__linux__)
    size_t pages = 0, rss = 0;
    std::ifstream("/proc/self/statm") >> pages >> rss;
    return rss * (size_t) sysconf(_SC_PAGESIZE) / 1024;
#else
    return 0;
#endif
}
int main() {
    llama_backend_init();
    size_t callbacks = 0;
    clip_context_params params{};
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval = [](ggml_tensor*, bool, void* p) { ++*(size_t*) p; return false; };
    params.cb_eval_user_data = &callbacks;
    clip_ctx ctx(params);
    ctx.ctx_data.reset(ggml_init({1 << 20, nullptr, true}));
    check(ctx.ctx_data != nullptr, "weight metadata context");
    ggml_tensor* weight = ggml_new_tensor_1d(ctx.ctx_data.get(), GGML_TYPE_F32, 1);
    ctx.buf.reset(ggml_backend_alloc_ctx_tensors(ctx.ctx_data.get(), ctx.backend_cpu));
    check(ctx.buf != nullptr, "persistent model-weight buffer");
    const float value = 3.5f;
    ggml_backend_tensor_set(weight, &value, 0, sizeof value);
    void* weight_address = weight->data;
    const size_t count = 16 * 1024 * 1024;  // 64 MiB input + 64 MiB output
    uint64_t expected_hash = 0;
    for (int round = 0; round < 3; ++round) {
        ggml_context_ptr graph_ctx(ggml_init({4 << 20, nullptr, true}));
        check(graph_ctx != nullptr, "graph metadata context");
        ggml_tensor* input = ggml_new_tensor_1d(graph_ctx.get(), GGML_TYPE_F32, count);
        ggml_set_input(input);
        ggml_tensor* output = ggml_add(graph_ctx.get(), input, weight);
        ggml_tensor* second = ggml_scale(graph_ctx.get(), input, 2.0f);
        ggml_set_output(output);
        ggml_set_output(second);
        auto* graph = ggml_new_graph_custom(graph_ctx.get(), 1024, false);
        ggml_build_forward_expand(graph, second);
        ggml_build_forward_expand(graph, output);
        check(ggml_backend_sched_alloc_graph(ctx.sched.get(), graph), "allocate real GGML CPU scratch");
        for (size_t i = 0; i < count; ++i) ((float*) input->data)[i] = (float) (i % 16) / 4;
        ggml_backend_cpu_set_n_threads(ctx.backend_cpu, 4);
        check(ggml_backend_sched_graph_compute(ctx.sched.get(), graph) == GGML_STATUS_SUCCESS, "compute actual GGML graph");
        uint64_t hash = 14695981039346656037ull;
        bool same = true;
        for (size_t i = 0; i < count; ++i) {
            const float wanted = (float) (i % 16) / 4 + value;
            same &= ((float*) output->data)[i] == wanted;
            same &= ((float*) second->data)[i] == ((float) (i % 16) / 4) * 2;
            uint32_t bits; std::memcpy(&bits, (float*) output->data + i, sizeof bits);
            hash = (hash ^ bits) * 1099511628211ull;
        }
        check(same, "every computed float matches");
        if (!round) expected_hash = hash;
        check(hash == expected_hash, "output bits repeat after scratch is rebuilt");
        ctx.is_allocated = true;
        const size_t before = rss_kib();
        const size_t freed = clip_release_cpu_work_buffers(&ctx);
#if defined(__linux__) && defined(__GLIBC__)
        malloc_trim(0);
#endif
        const size_t after = rss_kib();
        check(freed >= count * sizeof(float) * 2, "large graph scratch released");
        check(!ctx.is_allocated && weight->data == weight_address, "weights stay resident at the same address");
        check(*(float*) weight->data == value, "weight bits retained");
        check(ctx.flash_attn_type == CLIP_FLASH_ATTN_TYPE_DISABLED, "attention selection retained");
        check(clip_release_cpu_work_buffers(&ctx) == 0, "repeated release is a no-op");
        std::printf("workspace round %d: freed=%zu bytes RSS before=%zu KiB after=%zu KiB hash=%016llx\n",
                    round, freed, before, after, (unsigned long long) hash);
    }
    check(callbacks >= 3, "evaluation callback survives scheduler replacement");
    check(clip_release_cpu_work_buffers(nullptr) == 0, "null context");
    std::puts("CPU workspace integration: passed");
}
