// hipattr: what the HIP runtime says about host-native atomics, per device,
// from both the attribute query and the device-properties struct. The enum
// is resolved by the compiler, not parsed. Runs as its own process so that a
// failed query cannot leave a sticky error in the measuring process.
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  int n = 0;
  hipError_t e0 = hipGetDeviceCount(&n);
  if (e0 != hipSuccess) { printf("count_rc=%d\n", (int)e0); return 1; }
  for (int d = 0; d < n; d++) {
    int v = -1;
    hipError_t e1 = hipDeviceGetAttribute(&v, hipDeviceAttributeHostNativeAtomicSupported, d);
    hipDeviceProp_t p;
    hipError_t e2 = hipGetDeviceProperties(&p, d);
    printf("device=%d attr_rc=%d attr=%d props_rc=%d props=%d name=%s arch=%s\n",
           d, (int)e1, v, (int)e2, (e2 == hipSuccess ? p.hostNativeAtomicSupported : -1),
           (e2 == hipSuccess ? p.name : "?"), (e2 == hipSuccess ? p.gcnArchName : "?"));
  }
  printf("HIPATTR_DONE devices=%d\n", n);
  return 0;
}
