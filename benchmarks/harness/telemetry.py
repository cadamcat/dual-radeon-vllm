"""One telemetry shape for every machine this repository measures on.

Why this exists
---------------
Before 2026-09-02 the two runner families recorded disjoint sets. The Radeon
runners sampled per-card power and VRAM from sysfs and recorded no wall clock;
the CUDA runners recorded `wall_s` and `gen_tokens` and sampled no hardware at
all. Neither was a superset of the other, so no question could be asked across
machines without first asking which campaign it came from.

Worse, neither sampled clocks. A Colab T4 measured on 2026-09-02 ran a 400-step
matmul at 1305 MHz against a 1590 MHz ceiling while drawing 71.6 W against a
70 W cap -- power-throttled throughout. Nothing in either schema could have
distinguished that from a slow kernel.

What every field here is, and is not
------------------------------------
Every field below was probed on the actual machines on 2026-09-02 and returns a
real value on at least one platform. Fields that returned nothing anywhere are
absent by decision, and `ABSENT` records why, because "we did not record it" and
"it cannot be recorded" are different facts and the next person will ask.

  achieved memory bandwidth
      Not available on gfx1100 here. rocprofv3 runs in the VFIO guest and its
      kernel trace is correct -- names, grid sizes and durations all land -- but
      the memory-traffic counters read zero: FETCH_SIZE, GRBM_COUNT, GPUBusy and
      SQ_INSTS_VALU all summed to 0.0 over six dispatches of a 2048^2 fp32
      matmul, while SQ_WAVES on the same run returned 6120. Some counter blocks
      survive passthrough and the memory-side ones do not. On CUDA it is
      obtainable, but only under `ncu` on a microbenchmark: `dram__bytes.sum`
      returned 18.71 MB on a Colab T4. It is not obtainable during a served run
      on either platform, which is what this harness measures, so the
      utilisation figures this repository publishes stay derived from
      tok/s x bytes/token and are not cross-checked against hardware.

  PCIe throughput
      NVML gives it (`nvmlDeviceGetPcieThroughput`, 350/300 KB/s idle on a T4).
      amdgpu does not expose `pcie_bw` on Navi 31 -- the node is absent on both
      cards. Recorded where it exists, null where it does not, rather than
      dropped from the schema, because the asymmetry is a fact about the
      machines and hiding it would invite a cross-machine comparison that cannot
      be made.

  a second PMC counter in one pass
      rocprofv3 aborts with SIGABRT when given two `--pmc` counters on this box,
      at both 512^2 and 2048^2. One counter per pass works at both sizes. The
      axis is the counter count, not the workload.

Usage
-----
    from harness.telemetry import Sampler, describe

    s = Sampler(); s.start()
    ... run one cell ...
    row.update(s.stop_and_summarise())
    meta = describe()          # once per configuration
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time

SCHEMA_VERSION = 1

#: what this harness deliberately does not carry, and why. Kept as data so a
#: reader and a gate see the same list.
ABSENT = {
    "achieved_mem_bw_gbs":
        "gfx1100: rocprofv3 memory counters read zero under VFIO passthrough "
        "(FETCH_SIZE summed 0.0 while SQ_WAVES returned 6120). CUDA: only via "
        "ncu on a microbenchmark, not during a served run.",
    "pcie_bw_radeon":
        "amdgpu exposes no `pcie_bw` node on Navi 31; NVML has the equivalent, "
        "so the field is present and null on the Radeon side.",
}


def _f(path, cast=int, default=None):
    try:
        with open(path) as fh:
            return cast(fh.read().strip())
    except Exception:
        return default


def _dpm_current(path):
    """The starred entry of a pp_dpm_* table, in MHz, and the table's ceiling.

    The file looks like `0: 96Mhz *\n1: 456Mhz\n...`; the star is the state the
    card is in right now, and the last row is the highest state it may enter.
    """
    try:
        cur = top = None
        for line in open(path):
            m = re.search(r"(\d+)Mhz", line)
            if not m:
                continue
            v = int(m.group(1))
            top = v if top is None else max(top, v)
            if "*" in line:
                cur = v
        return cur, top
    except Exception:
        return None, None


class _AmdCard:
    """One passed-through Radeon, read entirely from sysfs."""

    def __init__(self, dev):
        self.dev = dev
        self.hwmon = next(iter(glob.glob(os.path.join(dev, "hwmon", "hwmon*"))), None)

    def slot(self):
        try:
            for line in open(os.path.join(self.dev, "uevent")):
                if line.startswith("PCI_SLOT_NAME="):
                    return line.strip().split("=", 1)[1]
        except Exception:
            pass
        return None

    def sample(self):
        s, s_top = _dpm_current(os.path.join(self.dev, "pp_dpm_sclk"))
        m, m_top = _dpm_current(os.path.join(self.dev, "pp_dpm_mclk"))
        r = {
            "gpu_busy_pct": _f(os.path.join(self.dev, "gpu_busy_percent")),
            "mem_busy_pct": _f(os.path.join(self.dev, "mem_busy_percent")),
            "vram_used_b": _f(os.path.join(self.dev, "mem_info_vram_used")),
            "sclk_mhz": s, "mclk_mhz": m,
            "sclk_mhz_cap": s_top, "mclk_mhz_cap": m_top,
            "pcie_tx_kbs": None, "pcie_rx_kbs": None,   # no pcie_bw on Navi 31
        }
        if self.hwmon:
            r["power_w"] = (_f(os.path.join(self.hwmon, "power1_average"), int, 0) or 0) / 1e6
            r["power_cap_w"] = (_f(os.path.join(self.hwmon, "power1_cap"), int, 0) or 0) / 1e6
            r["temp_c"] = (_f(os.path.join(self.hwmon, "temp1_input"), int, 0) or 0) / 1e3
        return r

    def static(self):
        return {"slot": self.slot(),
                "vram_total_b": _f(os.path.join(self.dev, "mem_info_vram_total")),
                "link_speed": _f(os.path.join(self.dev, "current_link_speed"), str),
                "link_width": _f(os.path.join(self.dev, "current_link_width"), str)}


class _NvCard:
    """One NVIDIA device, read through NVML in-process.

    NVML costs 9.2 ms for a full sample against 29.8 ms for one `nvidia-smi`
    subprocess, measured on a Colab T4, so the sampler never shells out.
    """

    def __init__(self, idx):
        import pynvml
        self.P = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def sample(self):
        P, h = self.P, self.h
        u = P.nvmlDeviceGetUtilizationRates(h)
        r = {"gpu_busy_pct": u.gpu, "mem_busy_pct": u.memory,
             "vram_used_b": P.nvmlDeviceGetMemoryInfo(h).used,
             "power_w": P.nvmlDeviceGetPowerUsage(h) / 1e3,
             "sclk_mhz": P.nvmlDeviceGetClockInfo(h, P.NVML_CLOCK_SM),
             "mclk_mhz": P.nvmlDeviceGetClockInfo(h, P.NVML_CLOCK_MEM)}
        for key, fn in (("sclk_mhz_cap", lambda: P.nvmlDeviceGetMaxClockInfo(h, P.NVML_CLOCK_SM)),
                        ("mclk_mhz_cap", lambda: P.nvmlDeviceGetMaxClockInfo(h, P.NVML_CLOCK_MEM)),
                        ("power_cap_w", lambda: P.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3),
                        ("temp_c", lambda: P.nvmlDeviceGetTemperature(h, P.NVML_TEMPERATURE_GPU)),
                        ("pcie_tx_kbs", lambda: P.nvmlDeviceGetPcieThroughput(
                            h, P.NVML_PCIE_UTIL_TX_BYTES)),
                        ("pcie_rx_kbs", lambda: P.nvmlDeviceGetPcieThroughput(
                            h, P.NVML_PCIE_UTIL_RX_BYTES))):
            try:
                r[key] = fn()
            except Exception:
                r[key] = None
        return r

    def static(self):
        P, h = self.P, self.h
        out = {"slot": None, "vram_total_b": P.nvmlDeviceGetMemoryInfo(h).total}
        for key, fn in (("link_speed", lambda: str(P.nvmlDeviceGetCurrPcieLinkGeneration(h))),
                        ("link_width", lambda: str(P.nvmlDeviceGetCurrPcieLinkWidth(h)))):
            try:
                out[key] = fn()
            except Exception:
                out[key] = None
        return out


def cards():
    """Every accelerator this machine will be measured on, in a stable order."""
    devs = sorted(d for d in glob.glob("/sys/class/drm/card*/device")
                  if os.path.exists(os.path.join(d, "gpu_busy_percent")))
    if devs:
        return [_AmdCard(d) for d in devs]
    try:
        import pynvml
        pynvml.nvmlInit()
        return [_NvCard(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        return []


class Sampler(threading.Thread):
    """Polls every card on a fixed cadence for the length of one cell.

    The first third and the last sixth of the samples are dropped, as the Radeon
    runners have always done: the head is engine warm-up and the tail is the
    request draining, and neither is the steady state the cell is about.
    """

    PERIOD_S = 1.5

    def __init__(self, period_s=None):
        super().__init__(daemon=True)
        self.period = period_s or self.PERIOD_S
        self.stop_ev = threading.Event()
        self.rows = []
        self.cards = cards()

    def run(self):
        while not self.stop_ev.is_set():
            try:
                self.rows.append([c.sample() for c in self.cards])
            except Exception:
                pass
            self.stop_ev.wait(self.period)

    def stop_and_summarise(self):
        self.stop_ev.set()
        self.join(timeout=self.period * 2 + 1)
        return summarise(self.rows)


def summarise(rows):
    """Per-cell aggregates, with the field names every machine emits."""
    if len(rows) >= 6:
        rows = rows[len(rows) // 3: -max(1, len(rows) // 6)]
    if not rows:
        return {"tele_samples": 0}
    n = len(rows[0])

    def vals(card, key):
        return [r[card].get(key) for r in rows if r[card].get(key) is not None]

    out = {"tele_samples": len(rows), "tele_schema": SCHEMA_VERSION}
    agg = {}
    for key, how in (("gpu_busy_pct", "max"), ("mem_busy_pct", "max"),
                     ("power_w", "max"), ("temp_c", "max"),
                     ("sclk_mhz", "max"), ("mclk_mhz", "max"),
                     ("vram_used_b", "max"),
                     ("pcie_tx_kbs", "max"), ("pcie_rx_kbs", "max")):
        per = []
        for c in range(n):
            v = vals(c, key)
            per.append(max(v) if v else None)
        agg[key] = per
        got = [x for x in per if x is not None]
        out[f"{key}_max"] = max(got) if got else None
    # power is the one quantity that is summed across cards rather than maxed:
    # a two-card box's draw is what the wall sees.
    psum = [sum(r[c].get("power_w") or 0 for c in range(n)) for r in rows]
    out["power_w_sum_max"] = round(max(psum), 1) if psum else None
    out["power_w_sum_min"] = round(min(psum), 1) if psum else None
    out["per_card"] = {k: v for k, v in agg.items()}
    # the guard rail the old schemas could not express
    caps_s = [rows[-1][c].get("sclk_mhz_cap") for c in range(n)]
    caps_p = [rows[-1][c].get("power_cap_w") for c in range(n)]
    out["sclk_mhz_cap"] = max([c for c in caps_s if c], default=None)
    out["power_cap_w"] = max([c for c in caps_p if c], default=None)
    if out["sclk_mhz_max"] and out["sclk_mhz_cap"]:
        out["sclk_pct_of_cap"] = round(out["sclk_mhz_max"] / out["sclk_mhz_cap"] * 100, 1)
    return out


def describe():
    """One-time machine description, written beside every campaign's rows."""
    cs = cards()
    return {"kind": "telemetry_meta", "tele_schema": SCHEMA_VERSION,
            "n_cards": len(cs),
            "cards": [c.static() for c in cs],
            "absent": ABSENT,
            "ts": time.time()}


if __name__ == "__main__":                       # smoke test on any machine
    s = Sampler(period_s=0.3)
    s.start()
    time.sleep(1.5)
    print(json.dumps({"describe": describe(), "sample": s.stop_and_summarise()},
                     indent=1, default=str))
