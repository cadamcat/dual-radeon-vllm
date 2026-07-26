// rccl_dumpstub.cpp — supplies the one symbol torch wants that RCCL 2.27.7 lacks.
//
// Only needed with the MULTI-ARCH build. The single-target (gfx1100) build in
// Releases already exports this symbol and needs no stub.
//
// PyTorch links `ncclCommDump`, an API that postdates RCCL 2.27.7. Two details
// cost us an evening, so they are worth stating plainly:
//
//   1. torch imports the C++ *mangled* name
//      (_Z12ncclCommDumpP8ncclCommRSt13unordered_map...), so an `extern "C"`
//      shim does not match and the loader still fails with an undefined symbol.
//   2. RCCL builds with -fvisibility=hidden, so a bare definition ends up LOCAL
//      and is invisible to the dynamic linker. Hence the explicit attribute.
//
// Build and attach:
//
//   clang++ -shared -fPIC -o librccl_dumpstub.so.1 rccl_dumpstub.cpp
//   patchelf --add-needed librccl_dumpstub.so.1 librccl.so.1
//
// The no-op return is safe: torch only calls this when dumping RCCL state for
// a debugger, which never happens on the inference path.

#include <unordered_map>
#include <string>

struct ncclComm;
typedef int ncclResult_t;

__attribute__((visibility("default")))
ncclResult_t ncclCommDump(ncclComm *, std::unordered_map<std::string, std::string> &) {
    return 0;   // ncclSuccess
}
