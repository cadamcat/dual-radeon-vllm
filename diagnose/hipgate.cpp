// hipgate.cpp — minimal probe for HIP 'operation cannot be performed in the
// present state' on VFIO guest, dual gfx1100. Models RCCL enqueue.cc:1750:
//   hipExtLaunchKernel(fn, grid, block, args, 0, stream, NULL, doneEvent, 0)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

static int failures = 0;
#define TRY(tag, cmd) do { \
  hipError_t e_ = (cmd); \
  printf("[%-22s] %-28s -> %s\n", tag, short_name(#cmd), hipGetErrorString(e_)); \
  fflush(stdout); \
  if (e_ != hipSuccess) failures++; \
} while(0)

static const char* short_name(const char* full) {
  static char buf[29];
  strncpy(buf, full, 28); buf[28] = 0;
  char* p = strchr(buf, '('); if (p) *p = 0;
  return buf;
}

__global__ void k_simple(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = p[i] * 2.f + 1.f;
}

// ~4KB by-value struct — NCCL passes kernelArgs as one big struct param
struct Big { char pad[4000]; float* out; };
__global__ void k_big(Big b) { *b.out = 42.f; }

int main(int argc, char** argv) {
  int nd = 0;
  TRY("init", hipGetDeviceCount(&nd));
  printf("devices=%d\n", nd);
  char tag[64];

  for (int d = 0; d < nd; d++) {
    snprintf(tag, 64, "dev%d.setdev", d);
    TRY(tag, hipSetDevice(d));
    float* buf = nullptr; int n = 1 << 20;
    snprintf(tag, 64, "dev%d.malloc", d);
    TRY(tag, hipMalloc(&buf, n * sizeof(float)));
    hipStream_t s, s2;
    snprintf(tag, 64, "dev%d.stream", d);
    TRY(tag, hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    TRY(tag, hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    hipEvent_t done, dep;
    snprintf(tag, 64, "dev%d.event", d);
    TRY(tag, hipEventCreateWithFlags(&done, hipEventDisableTiming));
    TRY(tag, hipEventCreateWithFlags(&dep, hipEventDisableTiming));

    void* args1[] = { &buf, &n };
    dim3 g(n / 256), b(256);

    // 1. plain launch
    snprintf(tag, 64, "dev%d.PLAIN", d);
    TRY(tag, hipLaunchKernel((void*)k_simple, g, b, args1, 0, s));
    TRY(tag, hipStreamSynchronize(s));

    // 2. RCCL exact shape
    snprintf(tag, 64, "dev%d.EXTLAUNCH", d);
    TRY(tag, hipExtLaunchKernel((void*)k_simple, g, b, args1, 0, s, NULL, done, 0));
    TRY(tag, hipStreamSynchronize(s));

    // 3. 4KB kernarg via ExtLaunch
    Big big; memset(&big, 0, sizeof big); big.out = buf;
    void* args2[] = { &big };
    snprintf(tag, 64, "dev%d.EXT-4KBARG", d);
    TRY(tag, hipExtLaunchKernel((void*)k_big, dim3(1), dim3(1), args2, 0, s, NULL, done, 0));
    TRY(tag, hipStreamSynchronize(s));

    // 4. cross-stream wait-event then ExtLaunch (ncclStrongStream analog)
    snprintf(tag, 64, "dev%d.XSTREAM", d);
    TRY(tag, hipEventRecord(dep, s2));
    TRY(tag, hipStreamWaitEvent(s, dep, 0));
    TRY(tag, hipExtLaunchKernel((void*)k_simple, g, b, args1, 0, s, NULL, done, 0));
    TRY(tag, hipStreamSynchronize(s));

    // 5. fine-grained coherent host memory + device kernel write (SHM analog)
    float* hostb = nullptr;
    snprintf(tag, 64, "dev%d.HOSTCOH", d);
    TRY(tag, hipHostMalloc((void**)&hostb, 4096, hipHostMallocMapped | hipHostMallocCoherent));
    if (hostb) {
      void* devptr = nullptr; int hn = 4096 / sizeof(float);
      TRY(tag, hipHostGetDevicePointer(&devptr, hostb, 0));
      void* args3[] = { &devptr, &hn };
      snprintf(tag, 64, "dev%d.EXT-HOSTMEM", d);
      TRY(tag, hipExtLaunchKernel((void*)k_simple, dim3(4), dim3(256), args3, 0, s, NULL, done, 0));
      TRY(tag, hipStreamSynchronize(s));
      hipHostFree(hostb);
    }
    hipFree(buf);
  }

  // capability probes (informational)
  for (int d = 0; d < nd; d++) {
    int vmm = -1, vmmRdma = -1, atomics = -1, pageable = -1;
    hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, d);
    hipDeviceGetAttribute(&vmmRdma, hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported, d);
    hipDeviceGetAttribute(&atomics, hipDeviceAttributeHostNativeAtomicSupported, d);
    hipDeviceGetAttribute(&pageable, hipDeviceAttributePageableMemoryAccess, d);
    printf("[dev%d.caps] VMM=%d VMM_RDMA=%d hostNativeAtomic=%d pageableAccess=%d\n",
           d, vmm, vmmRdma, atomics, pageable);
    for (int p = 0; p < nd; p++) {
      if (p == d) continue;
      int can = -1; hipDeviceCanAccessPeer(&can, d, p);
      printf("[dev%d.caps] canAccessPeer(%d->%d)=%d\n", d, d, p, can);
    }
  }

  printf(failures ? "RESULT: %d FAILURES\n" : "RESULT: ALL PASS\n", failures);
  return failures != 0;
}
