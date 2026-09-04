#!/usr/bin/env python3
"""Compare the collective RCCL sweep from B1's two NDEBUG arms.

The ratio printed by this tool is ``with-ndebug / without-ndebug``.  A ratio
above one therefore means that the NDEBUG arm is slower.  The comparator
requires matching B1 metadata before it compares the shared ``(hidden,
ntok)`` cells, and can optionally emit those per-cell comparisons as JSONL.

This tool compares the collective sweep only.  The end-to-end decode half of
B1 -- 2 models x 3 depths x 5 repeats -- is a separate measurement that this
tool does not touch; its shape is intentionally not encoded here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


TIMINGS = (
    "t_graph_us",
    "t_stream_us",
    "t_sync_us_median",
    "t_sync_us_min",
    "t_sync_us_p95",
)
HEADLINE_TIMINGS = ("t_graph_us", "t_stream_us")
SYNC_TIMINGS = (
    "t_sync_us_median",
    "t_sync_us_min",
    "t_sync_us_p95",
)
ENVIRONMENT_FIELDS = ("world_size", "dtype", "machine")
BASE_SYNTHETIC_KEYS = (
    (128, 1),
    (128, 2),
    (256, 1),
    (256, 2),
    (512, 1),
    (512, 2),
)


class B1Refusal(Exception):
    """A deliberate, named refusal with the status required by B1."""

    def __init__(self, name, status, detail):
        super().__init__(detail)
        self.name = name
        self.status = status
        self.detail = detail

    def display(self):
        return f"REFUSAL {self.name} (exit {self.status}): {self.detail}"


class Run:
    def __init__(self, path, meta, rccl_loaded, cells):
        self.path = Path(path)
        self.meta = meta
        self.rccl_loaded = rccl_loaded
        self.cells = cells


class Comparison:
    def __init__(self, with_run, without_run, rows):
        self.with_run = with_run
        self.without_run = without_run
        self.rows = rows
        self.with_only = sorted(set(with_run.cells) - set(without_run.cells))
        self.without_only = sorted(set(without_run.cells) - set(with_run.cells))


def _input_refusal(detail):
    return B1Refusal("B1-INPUT", 1, detail)


def _one_line(value):
    return str(value).replace("\n", "\\n")


def _require_number(record, field, path, line_number):
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _input_refusal(
            f"{path}:{line_number}: {field} must be a finite number"
        )
    if not math.isfinite(float(value)):
        raise _input_refusal(f"{path}:{line_number}: {field} is not finite")
    if value < 0:
        raise _input_refusal(f"{path}:{line_number}: {field} is negative")
    return value


def _require_integer(record, field, path, line_number):
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _input_refusal(f"{path}:{line_number}: {field} must be an integer")
    return value


def _read_run(path):
    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise _input_refusal(f"cannot read {path}: {exc}")

    meta_records = []
    cells = {}
    try:
        with handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise _input_refusal(
                        f"{path}:{line_number}: invalid JSON ({exc.msg})"
                    )
                if not isinstance(record, dict):
                    raise _input_refusal(
                        f"{path}:{line_number}: JSON record is not an object"
                    )

                kind = record.get("kind")
                if kind == "ar_meta":
                    meta_records.append(record)
                    continue
                if kind != "allreduce":
                    # telemetry_meta, ar_telemetry, ar_complete, and future
                    # non-sweep records are not per-cell measurements.
                    continue

                hidden = _require_integer(record, "hidden", path, line_number)
                ntok = _require_integer(record, "ntok", path, line_number)
                key = (hidden, ntok)
                if key in cells:
                    raise _input_refusal(
                        f"{path}:{line_number}: duplicate allreduce cell {key}"
                    )
                cells[key] = {
                    field: _require_number(record, field, path, line_number)
                    for field in TIMINGS
                }
    except OSError as exc:
        raise _input_refusal(f"cannot read {path}: {exc}")

    if len(meta_records) != 1:
        raise _input_refusal(
            f"{path}: expected exactly one ar_meta record, found {len(meta_records)}"
        )
    meta = meta_records[0]
    for field in ENVIRONMENT_FIELDS:
        if field not in meta:
            raise _input_refusal(f"{path}: ar_meta is missing {field}")

    rccl_loaded = meta.get("rccl_loaded")
    if not isinstance(rccl_loaded, dict):
        raise _input_refusal(f"{path}: ar_meta.rccl_loaded must be an object")
    for field in ("realpath", "md5", "version_string", "hidden_hostcall_buffer"):
        if field not in rccl_loaded:
            raise _input_refusal(f"{path}: ar_meta.rccl_loaded is missing {field}")
    if not isinstance(rccl_loaded["md5"], str) or not rccl_loaded["md5"]:
        raise _input_refusal(f"{path}: ar_meta.rccl_loaded.md5 must be non-empty text")
    if not isinstance(rccl_loaded["version_string"], str):
        raise _input_refusal(
            f"{path}: ar_meta.rccl_loaded.version_string must be text"
        )
    hostcall = rccl_loaded["hidden_hostcall_buffer"]
    if isinstance(hostcall, bool) or not isinstance(hostcall, int):
        raise _input_refusal(
            f"{path}: ar_meta.rccl_loaded.hidden_hostcall_buffer must be an integer"
        )
    if not cells:
        raise _input_refusal(f"{path}: no allreduce cells found")

    return Run(path, meta, rccl_loaded, cells)


def _compare_runs(with_path, without_path):
    with_run = _read_run(with_path)
    without_run = _read_run(without_path)

    # Check the direct build/version confound first.  It is more fundamental
    # than any later identity or arm-direction check: comparing 2.27.7 to
    # 2.30.4 changes two variables and is not a B1 experiment.
    if with_run.rccl_loaded["version_string"] != without_run.rccl_loaded[
        "version_string"
    ]:
        raise B1Refusal(
            "B1-VERSION",
            2,
            "version_string differs between arms; B1 needs the same RCCL "
            f"version (with-ndebug={_one_line(with_run.rccl_loaded['version_string'])!r}, "
            f"without-ndebug={_one_line(without_run.rccl_loaded['version_string'])!r})",
        )

    if with_run.rccl_loaded["md5"].lower() == without_run.rccl_loaded["md5"].lower():
        raise B1Refusal(
            "B1-MD5",
            3,
            "the two md5 values are equal; this is the same library twice",
        )

    with_hostcall = with_run.rccl_loaded["hidden_hostcall_buffer"]
    without_hostcall = without_run.rccl_loaded["hidden_hostcall_buffer"]
    if with_hostcall != 0 or without_hostcall == 0:
        raise B1Refusal(
            "B1-HOSTCALL-DIRECTION",
            4,
            "wrong arm direction or a build did not take: "
            f"with-ndebug hidden_hostcall_buffer={with_hostcall!r} (expected 0), "
            f"without-ndebug={without_hostcall!r} (expected non-zero)",
        )

    mismatches = []
    for field in ENVIRONMENT_FIELDS:
        with_value = with_run.meta[field]
        without_value = without_run.meta[field]
        if with_value != without_value:
            mismatches.append(
                f"{field}: with-ndebug={with_value!r}, "
                f"without-ndebug={without_value!r}"
            )
    if mismatches:
        raise B1Refusal(
            "B1-ENVIRONMENT",
            5,
            "world_size, dtype, and machine must match; " + "; ".join(mismatches),
        )

    shared = sorted(set(with_run.cells) & set(without_run.cells))
    if len(shared) < 5:
        raise B1Refusal(
            "B1-SHARED-CELLS",
            6,
            f"only {len(shared)} shared (hidden, ntok) cells; at least 5 are required",
        )

    rows = []
    for key in shared:
        with_values = with_run.cells[key]
        without_values = without_run.cells[key]
        ratios = {}
        for field in TIMINGS:
            denominator = without_values[field]
            if denominator <= 0:
                raise _input_refusal(
                    f"{without_run.path}: cell {key} has non-positive "
                    f"{field}, cannot calculate a ratio"
                )
            ratios[field] = with_values[field] / denominator
        rows.append(
            {
                "hidden": key[0],
                "ntok": key[1],
                "with_ndebug": copy.deepcopy(with_values),
                "without_ndebug": copy.deepcopy(without_values),
                "ratios": ratios,
            }
        )
    return Comparison(with_run, without_run, rows)


def _fmt(value):
    return f"{value:.3f}"


def _print_comparison(comparison):
    with_run = comparison.with_run
    without_run = comparison.without_run
    print(
        "B1 RCCL collective sweep "
        "(ratio = with-ndebug / without-ndebug; lower latency wins)"
    )
    print(
        "metadata: "
        f"version={_one_line(with_run.rccl_loaded['version_string'])!r}; "
        f"world_size={with_run.meta['world_size']!r}; "
        f"dtype={with_run.meta['dtype']!r}; machine={with_run.meta['machine']!r}"
    )
    print(
        "libraries: "
        f"with-ndebug md5={with_run.rccl_loaded['md5']} "
        f"hostcall={with_run.rccl_loaded['hidden_hostcall_buffer']}; "
        f"without-ndebug md5={without_run.rccl_loaded['md5']} "
        f"hostcall={without_run.rccl_loaded['hidden_hostcall_buffer']}"
    )
    print()

    headings = (
        f"{'hidden':>6} {'ntok':>5} | "
        f"{'graph with(us)':>15} {'graph without(us)':>18} {'graph ratio':>12} | "
        f"{'stream with(us)':>15} {'stream without(us)':>19} {'stream ratio':>13}"
    )
    print(headings)
    print("-" * len(headings))
    for row in comparison.rows:
        with_values = row["with_ndebug"]
        without_values = row["without_ndebug"]
        ratios = row["ratios"]
        print(
            f"{row['hidden']:6d} {row['ntok']:5d} | "
            f"{_fmt(with_values['t_graph_us']):>15} "
            f"{_fmt(without_values['t_graph_us']):>18} "
            f"{_fmt(ratios['t_graph_us']):>12} | "
            f"{_fmt(with_values['t_stream_us']):>15} "
            f"{_fmt(without_values['t_stream_us']):>19} "
            f"{_fmt(ratios['t_stream_us']):>13}"
        )

    sync_headings = (
        f"{'hidden':>6} {'ntok':>5} | "
        f"{'sync median with':>17} {'sync median without':>20} {'ratio':>8} | "
        f"{'sync min with':>14} {'sync min without':>17} {'ratio':>8} | "
        f"{'sync p95 with':>14} {'sync p95 without':>17} {'ratio':>8}"
    )
    print("\nt_sync_us_* (reported separately; ratios are with-ndebug / without-ndebug)")
    print(sync_headings)
    print("-" * len(sync_headings))
    for row in comparison.rows:
        with_values = row["with_ndebug"]
        without_values = row["without_ndebug"]
        ratios = row["ratios"]
        print(
            f"{row['hidden']:6d} {row['ntok']:5d} | "
            f"{_fmt(with_values['t_sync_us_median']):>17} "
            f"{_fmt(without_values['t_sync_us_median']):>20} "
            f"{_fmt(ratios['t_sync_us_median']):>8} | "
            f"{_fmt(with_values['t_sync_us_min']):>14} "
            f"{_fmt(without_values['t_sync_us_min']):>17} "
            f"{_fmt(ratios['t_sync_us_min']):>8} | "
            f"{_fmt(with_values['t_sync_us_p95']):>14} "
            f"{_fmt(without_values['t_sync_us_p95']):>17} "
            f"{_fmt(ratios['t_sync_us_p95']):>8}"
        )

    print("\nSummary")
    print(f"  cells compared: {len(comparison.rows)}")
    print(
        "  unshared cells: "
        f"total={len(comparison.with_only) + len(comparison.without_only)} "
        f"(with-ndebug-only={len(comparison.with_only)}, "
        f"without-ndebug-only={len(comparison.without_only)})"
    )
    for field in HEADLINE_TIMINGS:
        ratios = [row["ratios"][field] for row in comparison.rows]
        with_wins = sum(
            row["with_ndebug"][field] < row["without_ndebug"][field]
            for row in comparison.rows
        )
        without_wins = sum(
            row["without_ndebug"][field] < row["with_ndebug"][field]
            for row in comparison.rows
        )
        ties = len(ratios) - with_wins - without_wins
        print(
            f"  {field} ratios (with/without): "
            f"worst with-ndebug slower={_fmt(max(ratios))}, "
            f"worst without-ndebug slower={_fmt(max(1.0 / ratio for ratio in ratios))}, "
            f"median={_fmt(statistics.median(ratios))}"
        )
        print(
            f"  {field} wins (lower latency): "
            f"with-ndebug={with_wins}, without-ndebug={without_wins}, ties={ties}"
        )


def _write_jsonl(path, comparison):
    path = Path(path)
    try:
        with path.open("w", encoding="utf-8") as handle:
            for row in comparison.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as exc:
        raise B1Refusal("B1-OUTPUT", 1, f"cannot write {path}: {exc}")


def _synthetic_records(
    arm,
    keys,
    *,
    version="RCCL version 2.27.7",
    md5=None,
    hostcall=None,
    world_size=2,
    dtype="torch.bfloat16",
    machine="synthetic-atomics",
    slowdown_cells=(),
    slowdown_arm=None,
):
    if md5 is None:
        md5 = "synthetic-with-md5" if arm == "with" else "synthetic-without-md5"
    if hostcall is None:
        hostcall = 0 if arm == "with" else 32
    slowdown_cells = set(slowdown_cells)
    records = [
        {
            "kind": "ar_meta",
            "ts": 1.0,
            "machine": machine,
            "world_size": world_size,
            "dtype": dtype,
            "torch": "synthetic",
            "hip": "synthetic",
            "rccl_loaded": {
                "mapped": ["/synthetic/librccl.so"],
                "realpath": "/synthetic/librccl.so",
                "md5": md5,
                "version_string": version,
                "hidden_hostcall_buffer": hostcall,
            },
            "env": {},
            "device_names": ["synthetic-gpu", "synthetic-gpu"],
            "telemetry_import_error": None,
        },
        {
            "kind": "telemetry_meta",
            "tele_schema": 1,
            "n_cards": 2,
            "cards": [],
        },
    ]
    for hidden, ntok in keys:
        base = float(hidden) / 16.0 + float(ntok)
        values = {
            "t_graph_us": base + 10.0,
            "t_stream_us": base + 20.0,
            "t_sync_us_median": base + 30.0,
            "t_sync_us_min": base + 25.0,
            "t_sync_us_p95": base + 40.0,
        }
        if arm == slowdown_arm and (hidden, ntok) in slowdown_cells:
            values = {field: value * 1.12 for field, value in values.items()}
        records.append(
            {
                "kind": "allreduce",
                "rank": 0,
                "ts": 2.0 + len(records),
                "machine": machine,
                "hidden": hidden,
                "ntok": ntok,
                "bytes": hidden * ntok * 2,
                "iters": 10,
                "world_size": world_size,
                **values,
                "graph_error": None,
            }
        )
    records.extend(
        [
            {"kind": "ar_telemetry", "machine": machine, "tele_samples": 1},
            {"kind": "ar_complete", "ts": 99.0},
        ]
    )
    return records


def _write_records(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _make_synthetic_pair(directory, *, with_options=None, without_options=None,
                         with_keys=None, without_keys=None):
    with_options = dict(with_options or {})
    without_options = dict(without_options or {})
    if with_keys is None:
        with_keys = BASE_SYNTHETIC_KEYS
    if without_keys is None:
        without_keys = BASE_SYNTHETIC_KEYS
    with_path = Path(directory) / "with-ndebug.jsonl"
    without_path = Path(directory) / "without-ndebug.jsonl"
    _write_records(
        with_path,
        _synthetic_records("with", with_keys, **with_options),
    )
    _write_records(
        without_path,
        _synthetic_records("without", without_keys, **without_options),
    )
    return with_path, without_path


def _run_selftest_cli(with_path, without_path, output_path=None):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--with-ndebug",
        str(with_path),
        "--without-ndebug",
        str(without_path),
    ]
    if output_path is not None:
        command.extend(["--json", str(output_path)])
    try:
        return subprocess.run(
            command,
            cwd=str(Path(with_path).parent),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"could not run selftest child process: {exc}")


def _read_jsonl_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _selftest():
    failures = []

    def check(label, condition, detail=""):
        if condition:
            print(f"PASS {label}")
        else:
            message = f"FAIL {label}"
            if detail:
                message += f": {detail}"
            print(message)
            failures.append(message)

    displayed_synthetic_output = ""
    # The case directories are created inside one temporary directory.  This
    # helper keeps all inputs and output files outside the repository.
    try:
        with tempfile.TemporaryDirectory(prefix="compare-ndebug-selftest-") as temp:
            root = Path(temp)

            identical_dir = root / "identical"
            identical_dir.mkdir()
            with_path, without_path = _make_synthetic_pair(identical_dir)
            identical_json = identical_dir / "rows.jsonl"
            identical_proc = _run_selftest_cli(
                with_path, without_path, identical_json
            )
            identical_rows = _read_jsonl_rows(identical_json)
            check(
                "identical arms ratio 1.000 on every cell",
                identical_proc.returncode == 0
                and len(identical_rows) == len(BASE_SYNTHETIC_KEYS)
                and all(
                    row["ratios"][field] == 1.0
                    for row in identical_rows
                    for field in TIMINGS
                )
                and identical_proc.stdout.count("1.000") >= len(BASE_SYNTHETIC_KEYS),
                identical_proc.stdout + identical_proc.stderr,
            )

            slowdown_dir = root / "slowdown"
            slowdown_dir.mkdir()
            slowdown_cells = {(128, 2), (512, 1)}
            with_path, without_path = _make_synthetic_pair(
                slowdown_dir,
                with_options={
                    "slowdown_cells": slowdown_cells,
                    "slowdown_arm": "with",
                },
            )
            slowdown_json = slowdown_dir / "rows.jsonl"
            slowdown_proc = _run_selftest_cli(
                with_path, without_path, slowdown_json
            )
            slowdown_rows = _read_jsonl_rows(slowdown_json)
            slowdown_by_key = {
                (row["hidden"], row["ntok"]): row for row in slowdown_rows
            }
            slowdown_ok = slowdown_proc.returncode == 0
            for key in BASE_SYNTHETIC_KEYS:
                expected = 1.12 if key in slowdown_cells else 1.0
                row = slowdown_by_key.get(key)
                slowdown_ok = slowdown_ok and row is not None
                if row is not None:
                    slowdown_ok = slowdown_ok and math.isclose(
                        row["ratios"]["t_graph_us"], expected, rel_tol=1e-9
                    )
                    slowdown_ok = slowdown_ok and math.isclose(
                        row["ratios"]["t_stream_us"], expected, rel_tol=1e-9
                    )
            slowdown_ok = slowdown_ok and slowdown_proc.stdout.count("1.120") >= 2
            check(
                "seeded 12% slowdown is reported on the selected cells",
                slowdown_ok,
                slowdown_proc.stdout + slowdown_proc.stderr,
            )
            displayed_synthetic_output = slowdown_proc.stdout

            refusal_cases = [
                (
                    "version refusal has status 2",
                    "B1-VERSION",
                    2,
                    {"without_options": {"version": "RCCL version 2.30.4"}},
                ),
                (
                    "md5 refusal has status 3",
                    "B1-MD5",
                    3,
                    {
                        "with_options": {"md5": "same-md5"},
                        "without_options": {"md5": "same-md5"},
                    },
                ),
                (
                    "hostcall direction refusal has status 4",
                    "B1-HOSTCALL-DIRECTION",
                    4,
                    {"with_options": {"hostcall": 7}},
                ),
                (
                    "environment refusal has status 5",
                    "B1-ENVIRONMENT",
                    5,
                    {"without_options": {"machine": "different-machine"}},
                ),
            ]
            for index, (label, name, status, options) in enumerate(refusal_cases):
                case_dir = root / f"refusal-{index}"
                case_dir.mkdir()
                pair = _make_synthetic_pair(case_dir, **options)
                proc = _run_selftest_cli(*pair)
                combined = proc.stdout + proc.stderr
                check(
                    label,
                    proc.returncode == status
                    and f"REFUSAL {name} (exit {status})" in combined,
                    combined,
                )

            shared_dir = root / "shared-cells"
            shared_dir.mkdir()
            pair = _make_synthetic_pair(
                shared_dir, with_keys=BASE_SYNTHETIC_KEYS[:4]
            )
            proc = _run_selftest_cli(*pair)
            combined = proc.stdout + proc.stderr
            check(
                "fewer than five shared cells refusal has status 6",
                proc.returncode == 6
                and "REFUSAL B1-SHARED-CELLS (exit 6)" in combined,
                combined,
            )

            unshared_dir = root / "unshared"
            unshared_dir.mkdir()
            pair = _make_synthetic_pair(
                unshared_dir,
                with_keys=BASE_SYNTHETIC_KEYS + ((1024, 1),),
            )
            unshared_json = unshared_dir / "rows.jsonl"
            proc = _run_selftest_cli(pair[0], pair[1], unshared_json)
            combined = proc.stdout + proc.stderr
            unshared_rows = _read_jsonl_rows(unshared_json)
            check(
                "unshared cell is counted and excluded",
                proc.returncode == 0
                and len(unshared_rows) == len(BASE_SYNTHETIC_KEYS)
                and "unshared cells: total=1" in combined
                and "with-ndebug-only=1" in combined,
                combined,
            )
    except Exception as exc:  # pragma: no cover - defensive selftest reporting
        check("selftest execution", False, str(exc))

    if displayed_synthetic_output:
        print("\nSynthetic pair table (seeded 12% with-ndebug slowdown):")
        print(displayed_synthetic_output.rstrip())

    if failures:
        print(f"SELFTEST: FAIL ({len(failures)} case(s))", file=sys.stderr)
        return 1
    print("SELFTEST: PASS (8 cases)")
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compare RCCL collective sweeps built with and without NDEBUG."
    )
    parser.add_argument("--with-ndebug", metavar="JSONL")
    parser.add_argument("--without-ndebug", metavar="JSONL")
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="write one per-shared-cell comparison object per line",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the GPU-free synthetic test suite",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        if args.with_ndebug or args.without_ndebug or args.json:
            parser.error("--selftest cannot be combined with input or --json options")
        return _selftest()

    if not args.with_ndebug or not args.without_ndebug:
        parser.error(
            "normal comparison requires both --with-ndebug JSONL and "
            "--without-ndebug JSONL"
        )

    try:
        comparison = _compare_runs(args.with_ndebug, args.without_ndebug)
        if args.json:
            _write_jsonl(args.json, comparison)
        _print_comparison(comparison)
        return 0
    except B1Refusal as refusal:
        print(refusal.display(), file=sys.stderr)
        return refusal.status


if __name__ == "__main__":
    raise SystemExit(main())
