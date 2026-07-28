// repro-mmap-prot.hip.cpp — the same effect as repro-mmap-prot.py, with no
// PyTorch and no Python. A host->device copy out of a MAP_PRIVATE mapping is
// slow when the mapping is writable AND its pages are already resident, because
// KFD takes the fault permission from the VMA rather than from what the copy
// does: kfd_svm.c takes `readonly = !(vma->vm_flags & VM_WRITE)`, passes it to
// amdgpu_hmm_range_get_pages(), and amdgpu_hmm.c turns !readonly into
// HMM_PFN_REQ_WRITE, so copy-on-write is broken on every resident page even
// though the copy only reads.
//
// Exists so the case can be run where PyTorch is not installed — a hypervisor
// host, a rescue image, a machine with only the ROCm runtime.
//
//   hipcc -O2 --offload-arch=gfx1100 repro-mmap-prot.hip.cpp -o repro-mmap-prot
//   ./repro-mmap-prot [path]        # any filesystem: ext4, overlayfs and tmpfs
//                                   # all show it
//
// Verified against the Python version on the same guest, same file, same run:
// rw-p resident came out at 16 019.5 ms here against 16 019.3 ms there.
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static const size_t N    = 32ul << 20;   // bytes copied to the device
static const size_t SPAN = N * 8;        // bytes mapped

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) {                 \
    fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

static void make_file(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0 && (size_t)st.st_size >= SPAN) return;
    fprintf(stderr, "creating %zu MiB at %s\n", SPAN >> 20, path);
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); exit(1); }
    unsigned char *buf = (unsigned char *)malloc(1 << 20);
    for (int i = 0; i < (1 << 20); i++) buf[i] = (unsigned char)(i * 7 + 13);
    for (size_t w = 0; w < SPAN; w += (1 << 20)) fwrite(buf, 1, 1 << 20, f);
    free(buf);
    fclose(f);
}

static double run(const char *path, void *dev, bool writable, bool pretouch) {
    int fd = open(path, writable ? O_RDWR : O_RDONLY);
    if (fd < 0) { perror("open"); exit(1); }
    int prot = PROT_READ | (writable ? PROT_WRITE : 0);
    void *m = mmap(nullptr, SPAN, prot, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }

    if (pretouch) {                       // read the range the copy will use,
        volatile unsigned char sink = 0;  // which makes those pages resident
        const unsigned char *p = (const unsigned char *)m;
        for (size_t i = 0; i < N; i += 4096) sink ^= p[i];
        (void)sink;
    }

    CK(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    CK(hipMemcpy(dev, m, N, hipMemcpyHostToDevice));
    CK(hipDeviceSynchronize());
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

    munmap(m, SPAN);
    close(fd);
    return ms;
}

static void report(const char *tag, double ms) {
    printf("  %-22s %10.1f ms  %10.1f MiB/s\n", tag, ms, (N / (1024.0 * 1024.0)) / (ms / 1000.0));
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/var/tmp/repro-mmap-prot.bin";
    make_file(path);

    int fd = open(path, O_RDONLY);        // warm the page cache; not disk I/O
    if (fd >= 0) {
        unsigned char *b = (unsigned char *)malloc(1 << 24);
        while (read(fd, b, 1 << 24) > 0) {}
        free(b);
        close(fd);
    }

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device 0: %s, %zu MiB per copy, file %s\n\n", prop.gcnArchName, N >> 20, path);

    void *dev = nullptr;
    CK(hipMalloc(&dev, N));

    report("warm-up (r--p)",        run(path, dev, false, true));
    report("r--p, resident",        run(path, dev, false, true));
    report("rw-p, not resident",    run(path, dev, true,  false));
    report("rw-p, resident",        run(path, dev, true,  true));

    CK(hipFree(dev));
    return 0;
}
