// hipgate2.cpp — probe the AQL-dispatch 'present state' trigger.
// Prime suspect: dynamic shared memory (LDS) size in the ExtLaunch, the main
// thing an RCCL collective kernel has that hipgate.cpp's smem=0 launch lacked.
#include <hip/hip_runtime.h>
#include <cstdio>

extern __shared__ char dyn[];
__global__ void k_lds(float* p, int n, int smemBytes) {
  int i = threadIdx.x;
  if (i < smemBytes) dyn[i] = (char)i;      // touch dynamic LDS
  __syncthreads();
  if (i == 0 && p) p[0] = dyn[n % (smemBytes ? smemBytes : 1)];
}

int main() {
  int nd = 0; hipGetDeviceCount(&nd);
  printf("devices=%d\n", nd);
  int smems[] = {0, 8192, 16384, 32768, 49152, 65536, 66560};  // last two at/over 64KB LDS cap
  for (int d = 0; d < nd; d++) {
    hipSetDevice(d);
    int maxSmem = 0;
    hipDeviceGetAttribute(&maxSmem, hipDeviceAttributeMaxSharedMemoryPerBlock, d);
    printf("[dev%d] maxSharedMemPerBlock=%d bytes\n", d, maxSmem);
    float* buf = nullptr; hipMalloc(&buf, 4096);
    hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    hipEvent_t done; hipEventCreateWithFlags(&done, hipEventDisableTiming);
    for (int si = 0; si < (int)(sizeof(smems)/sizeof(int)); si++) {
      int smem = smems[si];
      int n = 32;
      void* args[] = { &buf, &n, &smem };
      // exact RCCL enqueue.cc:1750 shape, but WITH dynamic shared memory
      hipError_t e = hipExtLaunchKernel((void*)k_lds, dim3(1), dim3(256), args,
                                        (size_t)smem, s, NULL, done, 0);
      hipError_t es = (e == hipSuccess) ? hipStreamSynchronize(s) : e;
      printf("[dev%d] EXT smem=%6d -> launch:%s sync:%s\n", d, smem,
             hipGetErrorString(e), hipGetErrorString(es));
      fflush(stdout);
    }
    hipFree(buf);
  }
  return 0;
}
