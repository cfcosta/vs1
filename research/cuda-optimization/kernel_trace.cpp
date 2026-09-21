// Small CUPTI activity collector; kept outside the product build.
// Load with LD_PRELOAD and call vs1_trace_start/stop through dlsym.
#include <cupti.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct Sample { uint64_t ns = 0, count = 0; };
static std::map<std::string, Sample> samples;
static std::mutex lock;
static size_t dropped = 0;

static void check(CUptiResult result) {
    if (result != CUPTI_SUCCESS) {
        const char *message = nullptr;
        cuptiGetResultString(result, &message);
        std::fprintf(stderr, "CUPTI: %s\n", message);
        std::abort();
    }
}

static void CUPTIAPI request(uint8_t **buffer, size_t *size, size_t *records) {
    *size = 8 * 1024 * 1024;
    *buffer = static_cast<uint8_t *>(std::aligned_alloc(8, *size));
    if (!*buffer) std::abort();
    *records = 0;
}

static void CUPTIAPI complete(CUcontext context, uint32_t stream,
                             uint8_t *buffer, size_t, size_t valid) {
    std::lock_guard<std::mutex> guard(lock);
    CUpti_Activity *record = nullptr;
    while (true) {
        auto result = cuptiActivityGetNextRecord(buffer, valid, &record);
        if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        check(result);
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto *kernel = reinterpret_cast<CUpti_ActivityKernel9 *>(record);
            if (kernel->end < kernel->start || !kernel->start) std::abort();
            auto &sample = samples[kernel->name];
            sample.ns += kernel->end - kernel->start;
            sample.count++;
        }
    }
    size_t lost = 0;
    check(cuptiActivityGetNumDroppedRecords(context, stream, &lost));
    dropped += lost;
    std::free(buffer);
}

extern "C" void vs1_trace_start() {
    samples.clear();
    dropped = 0;
    check(cuptiActivityRegisterCallbacks(request, complete));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}

extern "C" void vs1_trace_stop(const char *path) {
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityFlushAll(0));
    std::lock_guard<std::mutex> guard(lock);
    auto *file = std::fopen(path, "w");
    if (!file) std::abort();
    std::vector<std::pair<std::string, Sample>> rows(samples.begin(), samples.end());
    std::sort(rows.begin(), rows.end(), [](auto &a, auto &b) { return a.second.ns > b.second.ns; });
    std::fprintf(file, "total_ns\tcount\tmean_ns\tkernel\n");
    for (auto &row : rows) {
        std::fprintf(file, "%llu\t%llu\t%.2f\t%s\n",
                     static_cast<unsigned long long>(row.second.ns),
                     static_cast<unsigned long long>(row.second.count),
                     double(row.second.ns) / row.second.count, row.first.c_str());
    }
    std::fclose(file);
    std::fprintf(stderr, "CUPTI: %zu kernel names, %zu dropped records\n", samples.size(), dropped);
    if (dropped || samples.empty()) std::abort();
}
