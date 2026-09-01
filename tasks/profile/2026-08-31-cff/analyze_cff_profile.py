#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


GAP_RE = re.compile(r"\[gap\]\s+\+\s*([0-9.]+) ms\s+@\s*([0-9.]+) ms\s+(.+)")
FLOAT_RE = r"([0-9.]+)"


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def parse_phase_log(path: Path) -> list[dict[str, float]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("[gap] ======== blake3 prove_fast"):
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current and any("[gap]" in line for line in current):
        blocks.append(current)

    parsed: list[dict[str, float]] = []
    for lines in blocks:
        marks: dict[str, tuple[float, float]] = {}
        text = "\n".join(lines)
        for line in lines:
            match = GAP_RE.search(line)
            if match:
                marks[match.group(3)] = (float(match.group(1)), float(match.group(2)))
        if "prover_data dropped" not in marks:
            continue

        def delta(label: str) -> float:
            return marks.get(label, (0.0, 0.0))[0]

        def cumulative(label: str) -> float:
            return marks.get(label, (0.0, 0.0))[1]

        metric = {
            "witness_ms": delta("witness: work done (incl. prefault)"),
            "seed_top_ms": delta("ntt: seed+top fused pass done"),
            "deep_ntt_merkle_ms": delta("ntt: deep pass done"),
            "commit_ms": delta("ntt: seed+top fused pass done") + delta("ntt: deep pass done"),
            "zc_round1_ms": delta("zc: round1 done (URM + C_s restore)"),
            "zc_round2_ms": delta("zc: round2 fused fold done"),
            "zc_n26_ms": delta("zc: rounds 3+4 composed fold done"),
            "zc_after_n26_ms": delta("zc: tail rounds done"),
            "zerocheck_total_ms": cumulative("zerocheck: pool exited")
            - cumulative("zerocheck: pool entered"),
            "lincheck_ms": delta("lincheck: done"),
            "open_ms": cumulative("open: returned") - cumulative("open: begin"),
            "total_profiled_ms": cumulative("prover_data dropped"),
        }
        patterns = {
            "lc_partial_fold_z_ms": rf"\[lc\] partial_fold_z\s+{FLOAT_RE} ms",
            "ligero_total_ms": rf"\[lig-prove\] total = {FLOAT_RE} ms",
            "ligero_initial_fold_ms": rf"initial sumcheck .*?: {FLOAT_RE} ms",
            "ligero_recursive_commits_ms": rf"recursive commits .*?:\s+{FLOAT_RE} ms",
            "ligero_induce_ms": rf"induce_sumcheck_poly:\s+{FLOAT_RE} ms",
            "zc_round1_ab_ms": rf"\[zc-timing\] round1 AB {FLOAT_RE} ms",
            "zc_round1_identity_c_ms": rf"identity-C fold {FLOAT_RE} ms",
            "zc_round2_reported_ms": rf"\[zc-timing\] round2 fused fold: {FLOAT_RE} ms",
            "zc_n24_ms": rf"n24:{FLOAT_RE}",
        }
        for key, pattern in patterns.items():
            found = re.search(pattern, text)
            if found:
                metric[key] = float(found.group(1))
        parsed.append(metric)
    return parsed


def summarize_phases(proofs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for proof in proofs for key in proof})
    return {
        key: {
            "median": median([proof[key] for proof in proofs if key in proof]),
            "min": min(proof[key] for proof in proofs if key in proof),
            "max": max(proof[key] for proof in proofs if key in proof),
        }
        for key in keys
    }


def parse_elapsed(root: Path) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for path in root.glob("cff-once-*/elapsed.txt"):
        text = path.read_text()
        mode = re.search(r"^mode=(.+)$", text, re.MULTILINE)
        elapsed = re.search(r"^elapsed_ns=(\d+)$", text, re.MULTILINE)
        if mode and elapsed:
            grouped[mode.group(1)].append(int(elapsed.group(1)) / 1_000_000)
    medians = {mode: median(values) for mode, values in grouped.items()}
    baseline = medians.get("off", 0.0)
    return {
        "runs_ms": dict(grouped),
        "median_ms": medians,
        "inflation_percent_vs_off": {
            mode: ((value / baseline) - 1.0) * 100.0 if baseline else 0.0
            for mode, value in medians.items()
        },
    }


def parse_scores(root: Path) -> dict[str, object]:
    result = {}
    for name, rel in {
        "clean": "cff-clean-profile-baseline/score.json",
        "traced": "cff-profile-trusted-run1/score.json",
    }.items():
        body = json.loads((root / rel).read_text())
        result[name] = {
            "score": body["score"],
            "median_seconds": body["metrics"]["median_seconds"],
            "p10_seconds": body["metrics"]["p10_seconds"],
            "verified": body["metrics"]["verified"],
            "measured_runs": body["metrics"]["measured_runs"],
        }
    clean = result["clean"]["median_seconds"]
    traced = result["traced"]["median_seconds"]
    result["trace_median_inflation_percent"] = (traced / clean - 1.0) * 100.0
    return result


def parse_counters(root: Path) -> dict[str, object]:
    events: dict[str, list[int]] = defaultdict(list)
    unsupported: set[str] = set()
    for path in root.glob("cff-perf-*/perf.csv"):
        for line in path.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            if len(fields) < 3:
                continue
            value, event = fields[0], fields[2]
            if value == "<not supported>":
                unsupported.add(event)
                continue
            try:
                events[event].append(int(float(value)))
            except ValueError:
                pass
    medians = {event: median(values) for event, values in sorted(events.items())}
    cycles = medians.get("cycles", 0.0)
    instructions = medians.get("instructions", 0.0)
    branches = medians.get("branches", 0.0)
    derived = {
        "ipc": instructions / cycles if cycles else 0.0,
        "branch_miss_percent": medians.get("branch-misses", 0.0) / branches * 100.0
        if branches
        else 0.0,
        "front_end_underdelivery_fraction": medians.get("idq_uops_not_delivered.core", 0.0)
        / (4.0 * cycles)
        if cycles
        else 0.0,
        "store_buffer_stall_cycles_fraction": medians.get("resource_stalls.sb", 0.0) / cycles
        if cycles
        else 0.0,
        "l1d_stall_cycles_fraction": medians.get("cycle_activity.stalls_l1d_miss", 0.0)
        / cycles
        if cycles
        else 0.0,
    }
    return {
        "event_runs": dict(events),
        "median": medians,
        "derived": derived,
        "unsupported": sorted(unsupported),
    }


def parse_hotspots(path: Path) -> dict[str, float]:
    categories = {
        "witness_drain": "Drain8>::drain_range_spread::<true>",
        "blake3_hash_many": "_blake3_hash_many_avx512",
        "zc_round2_nomat": "uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq",
        "ntt_fused4_prefetch": "butterfly_fused_4layer_row_pf::<1>",
        "zc_c4_gfni": "gfni_fold64_rows_masked_c4_bcast",
        "ntt_fused4": "butterfly_fused_4layer_row",
        "lincheck_gfni": "fold_block_major_gfni",
        "zc_fold2_plain": "fold2_plain_and_round_pair_lookahead_into",
        "ntt_sparse_dense_seed": "butterfly_fused_2layer_row_from_sparse_dense_geo",
        "pcs_fold16_banked": "fold16_banked",
    }
    totals = defaultdict(float)
    pattern = re.compile(r"^\s+([0-9.]+)%.*?\[\.\]\s+(.+)$")
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        percent = float(match.group(1))
        symbol = match.group(2)
        for category, needle in categories.items():
            if needle in symbol:
                if category == "ntt_fused4" and "row_pf" in symbol:
                    continue
                totals[category] += percent
                break
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    all_phase_blocks = parse_phase_log(root / "cff-profile-trusted.log")
    # The worker emits many setup/prewarm profile blocks before readiness.
    # The trusted verifier then runs one scored warm-up plus three measured
    # trials, which are the final four blocks in this append-only log.
    proofs = all_phase_blocks[-4:]
    result = {
        "phase_proofs": proofs,
        "phase_summary_ms": summarize_phases(proofs),
        "prewarm_phase_block_count": max(len(all_phase_blocks) - len(proofs), 0),
        "trace_overhead": parse_elapsed(root),
        "trusted_scores": parse_scores(root),
        "counters": parse_counters(root),
        "sampled_hotspots_percent": parse_hotspots(root / "cff-record-cycles1/report.txt"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
