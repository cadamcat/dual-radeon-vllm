#!/bin/bash
# preflight_host_link.sh -- record the HOST's PCIe link state for both cards,
# from the Proxmox host, into the campaign directory, before a run starts.
#
# Why from the host: the guest's sysfs reports the on-card bridge link, 16 GT/s
# x16 always, and so could not see that card 0b:00.0 spent 2026-08-29 onward at
# x8. The trained width is visible only at the host's root ports, three levels
# above the GPU (GPU -> on-card bridge downstream -> bridge upstream -> root port).
#
#   harness/preflight_host_link.sh <campaign-dir>      # from the Mac; uses `ssh pve`
#
# Writes <campaign-dir>/host_link.json. Exits 3 unless exactly two cards were
# read and both are x16, so a campaign cannot start on a narrowed link -- or on
# no reading at all -- without someone deciding to. The first version of this
# file passed on empty input, which is the one thing a preflight must not do.
set -eu
DIR=${1:?campaign directory}
HOST=${PVE_HOST:-pve}
RAW=$(ssh -o ConnectTimeout=20 "$HOST" '
for s in 0b:00.0 44:00.0; do
  p=$(readlink -f /sys/bus/pci/devices/0000:$s)
  rp=$(basename $(dirname $(dirname $(dirname $p))))
  sta=$(lspci -vv -s ${rp#0000:} | grep -m1 "LnkSta:" | sed "s/.*Speed //")
  printf "%s\t%s\t%s\n" "$s" "${rp#0000:}" "$sta"
done')
HL_RAW="$RAW" HL_DIR="$DIR" python3 - <<'PY'
import json, os, sys, time
rows = [l.split("\t") for l in os.environ["HL_RAW"].strip().splitlines() if l.strip()]
cards = [{"gpu": g, "root_port": rp, "lnksta": st,
          "width": st.split("Width ")[1].split()[0] if "Width " in st else None}
         for g, rp, st in rows]
out = {"kind": "host_link", "ts": time.time(),
       "read_from": "host root ports via lspci, three levels above the GPU",
       "cards": cards}
json.dump(out, open(os.environ["HL_DIR"].rstrip("/") + "/host_link.json", "w"), indent=1)
for c in cards:
    print(f"  {c['gpu']}  root {c['root_port']}  {c['lnksta']}")
if len(cards) != 2:
    print(f"REFUSING: read {len(cards)} card(s), expected 2 -- no reading is not a pass")
    sys.exit(3)
bad = [c for c in cards if c["width"] != "x16"]
if bad:
    print(f"REFUSING: {len(bad)} card(s) not at x16 -- fix the link, or record why in the README")
    sys.exit(3)
print("host link ok, both x16")
PY
