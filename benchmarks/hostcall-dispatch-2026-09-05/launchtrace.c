// launchtrace.so: LD_PRELOAD shim that prints the name of every kernel the
// process launches, via hipKernelNameRefByPtr / hipKernelNameRef, then calls
// the real launch. This wheel's HIP prints no ShaderName at any AMD_LOG_LEVEL
// and kineto records no GPU events here, so this is how the dispatched
// kernel is named. Output lines: "LAUNCH api=<api> name=<demangled or mangled>".
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <hip/hip_runtime_api.h>

typedef hipError_t (*launch_t)(const void*, dim3, dim3, void**, size_t, hipStream_t);
typedef hipError_t (*mlaunch_t)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                unsigned, hipStream_t, void**, void**);
typedef hipError_t (*extlaunch_t)(const void*, dim3, dim3, void**, size_t, hipStream_t, hipEvent_t, hipEvent_t, int);

hipError_t hipLaunchKernel(const void* f, dim3 g, dim3 b, void** args, size_t shm, hipStream_t s) {
  static launch_t real = 0;
  if (!real) real = (launch_t)dlsym(RTLD_NEXT, "hipLaunchKernel");
  const char* n = hipKernelNameRefByPtr(f, s);
  fprintf(stderr, "LAUNCH api=hipLaunchKernel name=%s\n", n ? n : "?"); fflush(stderr);
  return real(f, g, b, args, shm, s);
}
hipError_t hipExtLaunchKernel(const void* f, dim3 g, dim3 b, void** args, size_t shm, hipStream_t s,
                              hipEvent_t e0, hipEvent_t e1, int flags) {
  static extlaunch_t real = 0;
  if (!real) real = (extlaunch_t)dlsym(RTLD_NEXT, "hipExtLaunchKernel");
  const char* n = hipKernelNameRefByPtr(f, s);
  fprintf(stderr, "LAUNCH api=hipExtLaunchKernel name=%s\n", n ? n : "?"); fflush(stderr);
  return real(f, g, b, args, shm, s, e0, e1, flags);
}
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                                 unsigned bz, unsigned shm, hipStream_t s, void** args, void** extra) {
  static mlaunch_t real = 0;
  if (!real) real = (mlaunch_t)dlsym(RTLD_NEXT, "hipModuleLaunchKernel");
  const char* n = hipKernelNameRef(f);
  fprintf(stderr, "LAUNCH api=hipModuleLaunchKernel name=%s\n", n ? n : "?"); fflush(stderr);
  return real(f, gx, gy, gz, bx, by, bz, shm, s, args, extra);
}
