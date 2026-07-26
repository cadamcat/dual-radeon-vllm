// hipgate3.cpp — capstone: prove the hostcall requirement is the exact trigger.
// A kernel that calls device printf needs a hostcall buffer; hostcall needs PCIe
// atomics. If the printf kernel is refused while the plain kernel passes, the causal
// chain is nailed end to end, with no RCCL, no PyTorch and no second process.
//
// Reads four signals per kernel, because how the refusal surfaces varies by machine.
// In a VFIO guest (ROCm 7.14) hipExtLaunchKernel returns the error directly. On bare
// metal with a chipset-attached card (ROCm 7.2.4) launch and sync both return success
// and only hipGetLastError() reports it, while the device printf silently never
// arrives. Thanks to @adderek in ROCm/ROCm#6520 for that case; an earlier version of
// this probe, which read only the launch and sync return codes, would have called it
// a pass.
//
// Runs per device: one machine can have one affected GPU and one healthy one,
// depending on which lanes each card sits on.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k_plain(float* p) { if (threadIdx.x == 0 && p) p[0] = 1.f; }
__global__ void k_hostcall(float* p) {
  if (threadIdx.x == 0) { printf("    [device] HOSTCALL_MARKER reached the host\n"); if (p) p[0] = 2.f; }
}

static void run(const char* tag, const void* fn, float* buf, hipStream_t s, hipEvent_t ev) {
  (void)hipGetLastError();                       // clear anything already pending
  void* args[] = { &buf };
  hipError_t e  = hipExtLaunchKernel(fn, dim3(1), dim3(64), args, 0, s, NULL, ev, 0);
  hipError_t es = (e == hipSuccess) ? hipStreamSynchronize(s) : e;
  fflush(stdout);                                // let any device printf land first
  hipError_t el = hipGetLastError();
  const bool refused = (e != hipSuccess) || (es != hipSuccess) || (el != hipSuccess);
  printf("  %-9s %-8s  launch:%s | sync:%s | lastError:%s\n",
         tag, refused ? "REFUSED" : "ok",
         hipGetErrorString(e), hipGetErrorString(es), hipGetErrorString(el));
  fflush(stdout);
}

int main() {
  int nd = 0; hipGetDeviceCount(&nd);
  printf("devices=%d\n", nd);
  printf("affected = the hostcall kernel is REFUSED, or its marker never prints,\n"
         "           while the plain kernel is ok on the same device\n");
  for (int d = 0; d < nd; d++) {
    hipSetDevice(d);
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, d);
    float* buf = nullptr; hipMalloc(&buf, 256);
    hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    hipEvent_t ev; hipEventCreateWithFlags(&ev, hipEventDisableTiming);
    printf("\n--- device %d (%s) ---\n", d, prop.gcnArchName);
    run("plain",    (const void*)k_plain,    buf, s, ev);
    run("hostcall", (const void*)k_hostcall, buf, s, ev);
    hipEventDestroy(ev); hipStreamDestroy(s); hipFree(buf);
  }
  printf("\nIf no '[device] HOSTCALL_MARKER' line appeared for a device, hostcall is\n"
         "unavailable there, whatever the return codes said.\n");
  return 0;
}
