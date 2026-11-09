#include "strata/core/vision_records.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

static int checks = 0;
static void check(bool ok, const char* why) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    check(argc == 2, "pass a work directory");
    const fs::path file = fs::path(argv[1]) / ("vision-test-" + std::to_string(getpid()) + ".sve");
    strata::core::VisionRecords records;
    std::string err;
    // NaN, negative zero and a denormal must retain their exact float bits.
    const uint32_t bits[8] = {0x7fc01234, 0x80000000, 0x1, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0};
    auto write = [&](std::vector<int32_t> h, size_t payload, int copies = 1) {
        std::ofstream f(file, std::ios::binary | std::ios::trunc);
        for (int i = 0; i < copies; ++i) {
            f.write((char*) h.data(), (std::streamsize) h.size() * 4);
            f.write((char*) bits, (std::streamsize) payload);
        }
    };
    write({0x31455653, 2, 2, 1, 4}, sizeof bits, 2);
    check(records.open(file.string(), 4, 4, err), "map concatenated image records");
    check(records.images().size() == 2 && records.floats() == 16, "row counts and widths");
    for (const auto& im : records.images())
        check(std::memcmp(im.data, bits, sizeof bits) == 0, "embedding floats are bit-identical");
    fs::remove(file);
    check(std::memcmp(records.images()[1].data, bits, sizeof bits) == 0, "mapping remains valid after unlink");
    records.close();
    records.close();
    check(records.images().empty() && records.floats() == 0, "close is idempotent");
    check(!records.open(file.string(), 4, 4, err), "missing file");
    write({0x31455653, 2, 2, 1, 4}, sizeof bits, 2);
    check(!records.open(file.string(), 4, 3, err), "sum of image rows cannot exceed the prompt");
    check(records.images().empty() && records.floats() == 0, "failed open discards partial records");
    for (const std::vector<int32_t>& h : std::vector<std::vector<int32_t>>{
             {0, 2, 2, 1, 4}, {0x31455653, -1, 2, 1, 4}, {0x31455653, 2, 2, 2, 4},
             {0x31455653, 2, 2, 1, 3}, {0x31455653, 2, 2, 1, 0},
             {0x31455653, 0x7fffffff, 0x7fffffff, 1, 4}, {0x31455653, 2}}) {
        write(h, sizeof bits);
        check(!records.open(file.string(), 4, 4, err), "malformed record rejected before allocation");
        check(!err.empty() && records.images().empty(), "malformed record leaves no mapped state");
    }
    write({0x31455653, 2, 2, 1, 4}, sizeof bits - 1);
    check(!records.open(file.string(), 4, 4, err), "truncated payload");
    write({}, 0);
    check(!records.open(file.string(), 4, 4, err), "empty file");
    write({0x31455653, 2, 2, 1, 4}, sizeof bits);
    check(records.open(file.string(), 4, 2, err), "retry after malformed input succeeds");
    records.close();
    fs::remove(file);
    std::printf("mapped vision records: %d checks passed\n", checks);
}
