#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4.h"

#define GIB UINT64_C(1073741824)

static uint64_t add(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}

static uint64_t subtract(uint64_t a, uint64_t b) {
    return a > b ? a - b : 0;
}

static uint64_t budget(uint64_t physical, uint64_t metal,
                       uint64_t occupied, uint64_t reclaimable) {
    uint64_t available = subtract(physical, occupied);
    if (reclaimable < available) available = reclaimable;
    available = subtract(available, 2 * GIB);
    return metal < available ? metal : available;
}

static void self_test(void) {
    assert(add(UINT64_MAX, 1) == UINT64_MAX);
    assert(subtract(1, 2) == 0);
    assert(budget(128*GIB, 108*GIB, 16*GIB, 112*GIB) == 108*GIB);
    assert(budget(128*GIB, 120*GIB, 24*GIB, 104*GIB) == 102*GIB);
    assert(budget(128*GIB, 120*GIB, 16*GIB, 100*GIB) == 98*GIB);
    assert(budget(128*GIB, 120*GIB, 128*GIB, 0) == 0);
    assert(budget(128*GIB, 0, 0, 128*GIB) == 0);
    puts("GLM memory preflight arithmetic: PASS");
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        self_test();
        return 0;
    }
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf CONTEXT\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    errno = 0;
    unsigned long ctx = strtoul(argv[2], &end, 10);
    if (errno || end == argv[2] || *end || ctx == 0 || ctx > 1048576) {
        fprintf(stderr, "memory preflight: invalid GLM context\n");
        return 2;
    }
    @autoreleasepool {
        ds4_engine_options options = {0};
        options.model_path = argv[1];
        options.backend = DS4_BACKEND_METAL;
        options.inspect_only = true;
        options.power_percent = 100;
        options.context_size = (int)ctx;
        ds4_engine *engine = NULL;
        if (ds4_engine_open(&engine, &options) != 0) return 1;
        if (!ds4_engine_is_glm53(engine)) {
            fprintf(stderr, "memory preflight: expected GLM 5.3 GGUF\n");
            ds4_engine_close(engine);
            return 1;
        }
        const uint64_t weights = ds4_engine_model_bytes(engine);
        const ds4_context_memory graph =
            ds4_context_memory_estimate_with_prefill_mode(
                DS4_BACKEND_METAL, (int)ctx, 0, false);
        ds4_engine_close(engine);
        if (!weights || !graph.total_bytes || graph.comp_cap != ctx) {
            fprintf(stderr, "memory preflight: incomplete context estimate; refusing load\n");
            return 1;
        }
        uint64_t physical = 0;
        size_t size = sizeof(physical);
        vm_statistics64_data_t vm = {0};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_size_t page = 0;
        mach_port_t host = mach_host_self();
        kern_return_t page_rc = host_page_size(host, &page);
        kern_return_t stats_rc = host_statistics64(host, HOST_VM_INFO64,
                                                  (host_info64_t)&vm, &count);
        mach_port_deallocate(mach_task_self(), host);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (sysctlbyname("hw.memsize", &physical, &size, NULL, 0) != 0 ||
            page_rc != KERN_SUCCESS || stats_rc != KERN_SUCCESS || !page ||
            !physical || !device || !device.recommendedMaxWorkingSetSize) {
            fprintf(stderr, "memory preflight: cannot read host/Metal capacity; refusing load\n");
            return 1;
        }
        const uint64_t occupied = ((uint64_t)vm.wire_count +
            vm.internal_page_count + vm.compressor_page_count) * page;
        /* Free includes speculative pages; subtract them before adding file cache. */
        const uint64_t reclaimable = add(subtract(vm.free_count, vm.speculative_count),
                                        vm.external_page_count) * page;
        const uint64_t available = budget(physical, device.recommendedMaxWorkingSetSize,
                                          occupied, reclaimable);
        /* Planner omits some runtime allocations: observed Metal overhead ~1.5 GiB. */
        const uint64_t required = add(add(weights, graph.total_bytes), 2 * GIB);
        fprintf(stderr,
                "GLM memory preflight: ctx=%lu model=%.2f GiB graph=%.2f GiB "
                "+ runtime reserve=2.00 GiB => need=%.2f GiB\n"
                "  Metal limit=%.2f GiB; live budget=%.2f GiB "
                "(file cache reclaimable, 2 GiB host reserve)\n",
                ctx, (double)weights/GIB, (double)graph.total_bytes/GIB,
                (double)required/GIB, (double)device.recommendedMaxWorkingSetSize/GIB,
                (double)available/GIB);
        if (required > available) {
            fprintf(stderr, "Memory check FAILED: short %.2f GiB; close memory-heavy apps, "
                    "choose a smaller GGUF/context, or review the Metal wired limit. "
                    "Model not loaded.\n", (double)(required-available)/GIB);
            return 1;
        }
        fprintf(stderr, "Memory check PASS: %.2f GiB estimated headroom. "
                "Snapshot only; engine guards remain active.\n", (double)(available-required)/GIB);
    }
    return 0;
}
