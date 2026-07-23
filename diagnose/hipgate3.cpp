// hipgate3.cpp — capstone: prove hostcall requirement is the exact trigger.
// A kernel that calls device printf needs a hostcall buffer; hostcall needs
// PCIe atomics. If the printf kernel fails AQL dispatch with IllegalState
// while the plain kernel passes, the causal chain is nailed end to end.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k_plain(float* p) { if (threadIdx.x == 0 && p) p[0] = 1.f; }
__global__ void k_hostcall(float* p) {
  if (threadIdx.x == 0) { printf(""); if (p) p[0] = 2.f; }  // device printf -> hostcall
}

static void launch(const char* tag, const void* fn, float* buf, hipStream_t s, hipEvent_t ev) {
  void* args[] = { &buf };
  hipError_t e = hipExtLaunchKernel(fn, dim3(1), dim3(64), args, 0, s, NULL, ev, 0);
  hipError_t es = (e == hipSuccess) ? hipStreamSynchronize(s) : e;
  printf("[%s] launch:%-22s sync:%-22s\n", tag, hipGetErrorString(e), hipGetErrorString(es));
  fflush(stdout);
}

int main() {
  int nd = 0; hipGetDeviceCount(&nd);
  printf("devices=%d\n", nd);
  for (int d = 0; d < nd; d++) {
    hipSetDevice(d);
    float* buf = nullptr; hipMalloc(&buf, 256);
    hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    hipEvent_t ev; hipEventCreateWithFlags(&ev, hipEventDisableTiming);
    printf("--- dev%d ---\n", d);
    launch("plain    ", (const void*)k_plain,    buf, s, ev);
    launch("hostcall ", (const void*)k_hostcall, buf, s, ev);
    hipFree(buf);
  }
  return 0;
}
