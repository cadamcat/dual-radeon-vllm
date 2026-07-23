/* rsmi_stub.c — no-op ROCm-SMI shim. Lets the no-hostcall librccl resolve its
 * rsmi_* imports WITHOUT loading the real librocm_smi64 (which poisons torch's
 * amdsmi device enumeration -> device_count=0). All calls report failure so
 * RCCL falls back to its alt_rsmi (sysfs) path. Topology is irrelevant here
 * (NCCL_P2P_DISABLE=1, no P2P). */
#include <stdint.h>
typedef int rsmi_status_t;               /* 0 = success, non-zero = error */
#define ERR 8                            /* generic "not supported" */
#define WEAK __attribute__((weak))       /* real librocm_smi64 (strong) wins when loaded -> amdsmi unaffected */

WEAK rsmi_status_t rsmi_init(uint64_t flags) { (void)flags; return ERR; }
WEAK rsmi_status_t rsmi_status_string(rsmi_status_t s, const char **out) {
    (void)s; if (out) *out = "rsmi stubbed"; return 0;
}
WEAK rsmi_status_t rsmi_num_monitor_devices(uint32_t *n) { if (n) *n = 0; return ERR; }
WEAK rsmi_status_t rsmi_version_get(void *v) { (void)v; return ERR; }
WEAK rsmi_status_t rsmi_dev_firmware_version_get(uint32_t d, int b, uint64_t *o) {
    (void)d; (void)b; (void)o; return ERR;
}
WEAK rsmi_status_t rsmi_dev_pci_id_get(uint32_t d, uint64_t *o) { (void)d; (void)o; return ERR; }
WEAK rsmi_status_t rsmi_topo_get_link_type(uint32_t a, uint32_t b, uint64_t *h, int *t) {
    (void)a; (void)b; (void)h; (void)t; return ERR;
}
WEAK rsmi_status_t rsmi_topo_get_link_weight(uint32_t a, uint32_t b, uint64_t *w) {
    (void)a; (void)b; (void)w; return ERR;
}
WEAK rsmi_status_t rsmi_minmax_bandwidth_get(uint32_t a, uint32_t b, uint64_t *mn, uint64_t *mx) {
    (void)a; (void)b; (void)mn; (void)mx; return ERR;
}
