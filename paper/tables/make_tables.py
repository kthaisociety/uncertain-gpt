#!/usr/bin/env python3
"""Generate the paper's result tables from measured output. Nothing else.

    python3 paper/tables/make_tables.py            # from the repo root
    make -C paper tables                           # same thing

Inputs (both optional; whatever is missing becomes a placeholder):

  runs/results.csv   appended by scorecard.c, one row per training run.
                     Header: utc,arm,mode,cmd,steps,ms_per_step,vram_gb,
                     nll_first,nll_last,sigma_mean,sigma_min,sigma_max,gate
  runs/*.log         trainer/eval stdout. The uncertainty numbers are the
                     two "summary:" lines uq_report() prints in
                     gpt2_kernels.cuh, optionally preceded by the "eval:"
                     line uq_evaluate() prints.

Outputs: paper/tables/generated/tab-{cost,quality,uncertainty,ood,posterior,
bandwidth,amortization,fusion,crossover}.tex, each a bare `tabular` the document
wraps in a float. The last five are the cost tables and come from the `bench:`
and `bench-region:` lines; tab-cost is the training-run record from results.csv
and is generated but not currently displayed.

Those files are COMMITTED -- they are measurements, and CI has no runs/ to
remake them from -- so a run with nothing measured behind it leaves them alone
instead of overwriting a real number with a placeholder. Each carries a
`% generated:` stamp saying which commit and day its numbers came from; that is
how a stale one is spotted. See paper/README.md.

Splits: the quality and uncertainty tables carry one row per (arm, split),
never a preferred one. The off-distribution epistemic is NOT one of those
columns -- the OOD stream is a fixed synthetic sequence with no split
dependence, so it gets its own table and is stated once per arm.

THE ONE RULE: a cell is either a number that was measured or the macro
\\NA, which the preamble renders as a loud TODO. There is no third case,
and this script never computes a stand-in.
"""

import argparse
import csv
import datetime
import math
import os
import re
import subprocess
import sys
from glob import glob

NA = r"\NA"

# Provenance line prefix. It is the only part of a fragment allowed to change
# without the numbers changing, so it is also the only part ignored when
# deciding whether a rewrite is worth it.
STAMP = "% generated: "

# Canonical arms, in the order they appear in every table. Every arm gets a
# row whether or not it ran. Labels are the repo's own ("BBB", "Ens5",
# "LowR8"), so a table cell can be traced straight back to a log line; the
# prose is where they get their full names.
ARMS = ["BBB", "Vadam", "LowR", "Ens"]
ARM_ORDER = {k: i for i, k in enumerate(ARMS)}

# Both splits, always, in this order. Nothing here prefers one: which split
# the paper standardises on is an open decision, and a table that silently
# picked would resolve it by accident.
SPLITS = ["val", "test"]

# What the arms call themselves in eval output vs in the scorecard CSV.
EVAL_ALIAS = {"bbb": "BBB", "vadam": "Vadam", "lowr": "LowR", "ens": "Ens"}

MODE_ORDER = {"overfit": 0, "train": 1}


def canon(raw):
    """Map any observed arm label to (key, label). Unknown labels are kept
    verbatim and sorted last rather than silently dropped."""
    label = EVAL_ALIAS.get(raw.strip().lower(), raw.strip())
    for key in ARM_ORDER:
        if label.lower().startswith(key.lower()):
            return key, label
    return label, label


def fmt(value, digits):
    return NA if value is None else "%.*f" % (digits, value)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


# --------------------------------------------------------------------------
# runs/results.csv  ->  cost rows
# --------------------------------------------------------------------------
def read_csv_rows(path):
    """Latest row per (arm, mode); the file is appended to across sessions."""
    if not path or not os.path.exists(path):
        return {}
    latest = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            arm = (row.get("arm") or "").strip()
            mode = (row.get("mode") or "").strip()
            if not arm or not mode:
                continue
            k = (arm, mode)
            if k not in latest or (row.get("utc") or "") >= (latest[k].get("utc") or ""):
                latest[k] = row
    return latest


# --------------------------------------------------------------------------
# runs/*.log  ->  eval records
#
# Formats, from gpt2_kernels.cuh (uq_evaluate / uq_report):
#   eval: arm %-5s | split %s | S=%d... | %d batches x %d tokens = ...
#   summary: NLPD single %.4f -> %s %.4f | ECE(%d) %.4f | identity_err %.1e
#   summary: uncertainty (nats)  ID: total %.3f = alea %.3f + epi %.4f   OOD: epi %.4f  %s
# --------------------------------------------------------------------------
RE_EVAL = re.compile(r"^eval:\s+arm\s+(\S+)\s+\|\s+split\s+(.*?)\s+\|\s+S=(\d+)")
RE_NLPD = re.compile(
    r"^summary:\s+NLPD single\s+(\S+)\s+->\s+(\S+)\s+(\S+)\s+\|\s+"
    r"ECE\((\d+)\)\s+(\S+)\s+\|\s+identity_err\s+(\S+)")
RE_UNC = re.compile(
    r"^summary:\s+uncertainty \(nats\)\s+ID:\s+total\s+(\S+)\s+=\s+alea\s+(\S+)"
    r"\s+\+\s+epi\s+(\S+)\s+OOD:\s+epi\s+(\S+)")


def read_eval_records(paths):
    """One record per eval block. Blocks whose header line is absent (older
    logs) take their arm from the filename and leave split/S unknown."""
    records = []
    for path in paths:
        stem_arm = os.path.basename(path).split("_")[0]
        cur = None

        def flush():
            if cur and (cur.get("nlpd_mix") is not None or cur.get("total") is not None):
                records.append(cur)

        try:
            fh = open(path, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                m = RE_EVAL.match(line)
                if m:
                    flush()
                    cur = {"arm": m.group(1), "split": m.group(2),
                           "S": int(m.group(3)), "mtime": os.path.getmtime(path),
                           "src": path}
                    continue
                m = RE_NLPD.match(line)
                if m:
                    if cur is None:
                        cur = {"arm": stem_arm, "split": None, "S": None,
                               "mtime": os.path.getmtime(path), "src": path}
                    cur["nlpd_single"] = num(m.group(1))
                    cur["mix_label"] = m.group(2)
                    cur["nlpd_mix"] = num(m.group(3))
                    cur["ece_bins"] = int(m.group(4))
                    cur["ece"] = num(m.group(5))
                    continue
                m = RE_UNC.match(line)
                if m:
                    if cur is None:
                        cur = {"arm": stem_arm, "split": None, "S": None,
                               "mtime": os.path.getmtime(path), "src": path}
                    cur["total"] = num(m.group(1))
                    cur["alea"] = num(m.group(2))
                    cur["epi"] = num(m.group(3))
                    cur["epi_ood"] = num(m.group(4))
            flush()
    return records


# --------------------------------------------------------------------------
# runs/bench_*.log  ->  amortization records
#
# Format, from bench_report() in gpt2_kernels.cuh. One line per (arm, B):
#   bench: arm %s | B %d | T %d | tokens %d | params %zu | steps %d
#          | warmup %d | total %f ms | posterior %f ms | paramwork %f ms
#          | model %f ms | share %f | p25 %f | p75 %f
#          | temp %d-%d C | sm %d-%d MHz
#
# The variant tag is the filename: runs/bench_pre.log is before fusion,
# runs/bench_post.log after. Two variants of the same (arm, B) are what the
# fusion table subtracts, so they must not be merged by recency the way eval
# records are.
# --------------------------------------------------------------------------
RE_BENCH = re.compile(
    r"^bench:\s+arm\s+(\S+)\s+\|\s+B\s+(\d+)\s+\|\s+T\s+(\d+)\s+\|\s+tokens\s+(\d+)"
    r"\s+\|\s+params\s+(\d+)\s+\|\s+steps\s+(\d+)\s+\|\s+warmup\s+(\d+)"
    r"\s+\|\s+total\s+(\S+)\s+ms\s+\|\s+posterior\s+(\S+)\s+ms"
    r"\s+\|\s+paramwork\s+(\S+)\s+ms\s+\|\s+model\s+(\S+)\s+ms"
    r"\s+\|\s+share\s+(\S+)\s+\|\s+p25\s+(\S+)\s+\|\s+p75\s+(\S+)"
    r"\s+\|\s+temp\s+(-?\d+)-(-?\d+)\s+C\s+\|\s+sm\s+(-?\d+)-(-?\d+)\s+MHz")

BENCH_TAG = re.compile(r"^bench_(\w+)\.log$")

# One line per (arm, B, region), same files:
#   bench-region: arm %s | B %d | %-14s | %f ms
RE_REGION = re.compile(
    r"^bench-region:\s+arm\s+(\S+)\s+\|\s+B\s+(\d+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+ms")

# Parameter-sized buffer TOUCHES (reads + writes of a P-element buffer) per
# region, counted at the launch sites in the arm's .cu file. Structure, not
# measurement: this is the only thing in this file that is read off source
# rather than off a log, and it is here so the table can convert a measured
# time into an achieved bandwidth. The check that it is right is the table
# itself -- a wrong count puts its region off the band the others sit in.
TOUCHES = {
    ("pre", "bbb"): {"*sample": 3, "+zero_dw": 1, "*zero_post": 2,
                     "*post_bwd": 6, "*kl": 6, "+adamw_mu": 7,
                     "*adamw_rho": 7},
    ("post", "bbb"): {"*fused_step": 15},
}

# Bytes per element in every parameter-sized buffer: fp32 throughout.
BYTES_PER_ELEM = 4

# The arms that draw a weight vector. The control has no posterior region, so
# it has no row in the tables that price one.
BAYES_ARMS = ["BBB", "Vadam", "LowR"]

# The band inside which a difference is drift rather than a result. The control
# arm's code is identical in both sweeps, so what it reproduces of itself is the
# floor on what either sweep can resolve, and nothing smaller is separable from
# the card's thermal state. It is a constant here because the prose states it
# too; crossover_table() checks it against the control's own rows and says so on
# stderr if the sweeps ever drift wider than this.
DRIFT_BAND = 0.01


def read_bench_records(paths):
    """One record per (variant, arm, B). Later lines win within a file, so a
    re-run of one point can simply be appended."""
    out = {}
    for path in paths:
        m = BENCH_TAG.match(os.path.basename(path))
        if not m:
            continue
        tag = m.group(1)
        try:
            fh = open(path, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                b = RE_BENCH.match(line)
                if not b:
                    continue
                g = b.groups()
                out[(tag, canon(g[0])[0], int(g[1]))] = {
                    "tag": tag, "arm": g[0], "B": int(g[1]), "T": int(g[2]),
                    "tokens": int(g[3]), "params": int(g[4]),
                    "steps": int(g[5]), "warmup": int(g[6]),
                    "total": num(g[7]), "posterior": num(g[8]),
                    "paramwork": num(g[9]), "model": num(g[10]),
                    "share": num(g[11]), "p25": num(g[12]), "p75": num(g[13]),
                    "temp_lo": int(g[14]), "temp_hi": int(g[15]),
                    "sm_lo": int(g[16]), "sm_hi": int(g[17]),
                }
    return out


def read_region_records(paths):
    """Milliseconds per (variant, arm, B, region). Same files as the bench
    records; a re-run of one point simply overwrites it."""
    out = {}
    for path in paths:
        m = BENCH_TAG.match(os.path.basename(path))
        if not m:
            continue
        try:
            fh = open(path, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                r = RE_REGION.match(line)
                if r:
                    out[(m.group(1), r.group(1), int(r.group(2)),
                         r.group(3))] = num(r.group(4))
    return out


def pick(records, key, split):
    """Best record for an arm: preferred split first, then most recent."""
    cands = [r for r in records if canon(r["arm"])[0] == key]
    if not cands:
        return None
    same = [r for r in cands if r.get("split") == split]
    return max(same or cands, key=lambda r: r["mtime"])


def pick_split(records, key, split):
    """Most recent record for exactly this arm and this split, or None.
    Unlike pick() it never falls back: a val number must not be able to
    appear under a test label."""
    cands = [r for r in records
             if canon(r["arm"])[0] == key and r.get("split") == split]
    return max(cands, key=lambda r: r["mtime"]) if cands else None


def ratio(a, b):
    return None if not a or not b else a / b


# --------------------------------------------------------------------------
# emit
#
# Generated fragments are committed, which is what lets a checkout with no
# runs/ (CI) still build a PDF with real numbers in it. Three rules keep that
# safe; they are shared with make_figures.py, which imports them.
# --------------------------------------------------------------------------
def stamp(root):
    """The provenance line: which commit and which day these numbers came
    from. A committed fragment is otherwise indistinguishable from a fresh
    one, and this is the only way to tell a stale one."""
    try:
        p = subprocess.run(["git", "describe", "--always", "--dirty"],
                           cwd=root, capture_output=True, text=True, timeout=10)
        commit = p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        commit = ""
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return "%s%s on %s" % (STAMP, commit or "commit unknown", day)


def has_number(text):
    """True when some body cell of a fragment holds a measured value. Row
    labels (`BBB`, `val`, `PASS`) and \\NA are not values; a table with no
    number behind it must never overwrite a committed one."""
    rows = [ln for ln in text.splitlines() if ln.endswith(r"\\")]
    return any(re.match(r"-?\d", c.strip())
               for ln in rows[1:] for c in ln[:-2].split("&"))


def bare(text):
    """A fragment without its provenance line -- what "the same numbers"
    means when deciding whether a rewrite is worth it."""
    return [ln for ln in text.splitlines() if not ln.startswith(STAMP)]


def write_fragment(path, text, made, measured=None):
    """Write a generated fragment, stamped, with the two exceptions that
    make committing one safe. Returns what it did, for the caller's report.

      kept       nothing was measured and a fragment is already on disk --
                 keep its numbers rather than replace them with a TODO.
      unchanged  only the provenance stamp would move; rebuilding must not
                 churn the date on numbers that did not change.

    `measured` overrides the has_number() test for a fragment that carries
    no cells of its own (the figure's \\includegraphics).
    """
    lines = text.splitlines(True)
    stamped = "".join(lines[:1] + [made + "\n"] + lines[1:])
    old = None
    if os.path.exists(path):
        with open(path) as fh:
            old = fh.read()
    if old is not None:
        if not (has_number(text) if measured is None else measured):
            return "kept"
        if bare(old) == bare(stamped):
            return "unchanged"
    with open(path, "w") as fh:
        fh.write(stamped)
    return "wrote"


def table(header_cols, colspec, body_rows, note):
    out = ["% GENERATED by paper/tables/make_tables.py -- do not edit.",
           "%% %s" % note,
           r"\begin{tabular}{%s}" % colspec,
           r"\toprule",
           " & ".join(header_cols) + r" \\",
           r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in body_rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def cost_table(csv_rows, csv_path):
    body, seen = [], set()
    # Rows actually present in the CSV, in canonical arm order then mode.
    present = sorted(
        csv_rows.items(),
        key=lambda kv: (ARM_ORDER.get(canon(kv[0][0])[0], len(ARM_ORDER)),
                        canon(kv[0][0])[1], MODE_ORDER.get(kv[0][1], 9)))
    for (arm, mode), row in present:
        key, label = canon(arm)
        seen.add(key)
        body.append([
            tex_escape(label), tex_escape(mode),
            fmt(num(row.get("steps")), 0),
            fmt(num(row.get("ms_per_step")), 2),
            fmt(num(row.get("vram_gb")), 2),
            fmt(num(row.get("nll_first")), 4),
            fmt(num(row.get("nll_last")), 4),
            tex_escape(row.get("gate") or "") or NA,
        ])
    # An arm with no measurement at all still gets a visibly empty row.
    for key in ARMS:
        if key not in seen:
            body.append([tex_escape(key), NA] + [NA] * 6)
    return table(
        ["Arm", "Mode", "Steps", "ms/step", "VRAM (GB)",
         "NLL first", "NLL last", "Gate"],
        "llrrrrrl", body,
        "source: %s" % (csv_path if csv_path else "(no results.csv found)"))


def quality_table(records, srcs):
    body = []
    for key in ARMS:
        for split in SPLITS:
            r = pick_split(records, key, split) or {}
            body.append([
                tex_escape(canon(r["arm"])[1] if r.get("arm") else key),
                split,
                str(r["S"]) if r.get("S") else NA,
                fmt(r.get("nlpd_single"), 4),
                fmt(r.get("nlpd_mix"), 4),
                fmt(r.get("ece"), 4),
            ])
    return table(["Arm", "Split", "$S$", "NLPD (single draw)",
                  "NLPD (mixture)", "ECE"],
                 "llrrrr", body, "source: %s" % srcs)


def uncertainty_table(records, srcs):
    """In-distribution decomposition only. The off-distribution term is
    split-independent by construction and lives in ood_table()."""
    body = []
    for key in ARMS:
        for split in SPLITS:
            r = pick_split(records, key, split) or {}
            body.append([
                tex_escape(canon(r["arm"])[1] if r.get("arm") else key),
                split,
                str(r["S"]) if r.get("S") else NA,
                fmt(r.get("total"), 3),
                fmt(r.get("alea"), 3),
                fmt(r.get("epi"), 4),
            ])
    return table(["Arm", "Split", "$S$", "Total", "Aleatoric",
                  "Epistemic (ID)"],
                 "llrrrr", body, "source: %s" % srcs)


def ood_table(records, srcs):
    """One row per arm. The OOD stream is a fixed synthetic token sequence
    (a seed, not a split), so its epistemic term is one measurement, not one
    per split -- and the guard below refuses to state it once if the two
    splits ever stop agreeing."""
    body, disagreed = [], []
    for key in ARMS:
        rs = {s: pick_split(records, key, s) or {} for s in SPLITS}
        vals = {r["epi_ood"] for r in rs.values() if r.get("epi_ood") is not None}
        if len(vals) > 1:
            disagreed.append(key)
        epi_ood = vals.pop() if len(vals) == 1 else None
        Ss = {r["S"] for r in rs.values() if r.get("S")}
        label = next((canon(r["arm"])[1] for r in rs.values() if r.get("arm")), key)
        body.append(
            [tex_escape(label),
             str(Ss.pop()) if len(Ss) == 1 else NA,
             fmt(epi_ood, 4)]
            + [fmt(ratio(epi_ood, rs[s].get("epi")), 3) for s in SPLITS])
    return (table(["Arm", "$S$", "Epistemic (OOD)"]
                  + ["OOD/ID (%s)" % s for s in SPLITS],
                  "lrrrr", body, "source: %s" % srcs), disagreed)


def local_exponent(lo, hi):
    """k in share = B^-k between two adjacent measured points."""
    if not (lo and hi) or lo["share"] <= 0 or hi["share"] <= 0:
        return None
    return (math.log(lo["share"] / hi["share"])
            / math.log(float(hi["B"]) / float(lo["B"])))


def posterior_table(bench, tag, srcs):
    """The Bayesian work's own cost, per arm, over the whole batch sweep.

    The posterior column is DIRECTLY timed: before fusion each arm's Bayesian
    work sits in its own kernel launches, every launch is bracketed by CUDA
    events, and nothing here is a difference against a deterministic baseline
    that also differs elsewhere. That isolation is what fusion gives up, so
    this table is stated for the unfused variant and the fused one is priced
    in total O(P) work instead (see fusion_table).

    One row per arm rather than one per B: the low-rank arm's largest point is
    refused by the VRAM guard, and a per-B table would have to render that
    refusal as an unmeasured cell. Each arm's range is therefore its own, and
    the spread is over the points it reached."""
    body = []
    for key in BAYES_ARMS:
        rs = sorted((r for k, r in bench.items() if k[0] == tag and k[1] == key),
                    key=lambda r: r["B"])
        if not rs:
            body.append([key] + [NA] * 8)
            continue
        ms = [r["posterior"] for r in rs]
        lo, hi = min(ms), max(ms)
        body.append([
            tex_escape(canon(rs[0]["arm"])[1]),
            "%d--%d" % (rs[0]["B"], rs[-1]["B"]),
            fmt(lo, 3), fmt(hi, 3),
            fmt(100.0 * (hi / lo - 1.0), 1) + r"\%",
            fmt(100.0 * rs[0]["share"], 2) + r"\%",
            fmt(100.0 * rs[-1]["share"], 2) + r"\%",
            fmt(local_exponent(rs[0], rs[1]) if len(rs) > 1 else None, 3),
            fmt(local_exponent(rs[-2], rs[-1]) if len(rs) > 1 else None, 3),
        ])
    return table(
        ["Arm", "$B$ range", "Posterior ms (min)", "(max)", "Spread",
         "Share at $B=1$", "Share at max $B$", "$k$ (small end)",
         "$k$ (top end)"],
        "llrrrrrrr", body,
        "source: %s (variant %s)" % (srcs, tag))


def bandwidth_table(regions, bench, tag_before, tag_after, srcs, arm="bbb"):
    """Time per region converted to achieved bandwidth, which is what makes
    the touch counts a measurement rather than a claim: structurally unrelated
    regions have to land in one band, and the aggregate has to land on the
    card's own ceiling. A region off the band means its count is wrong.

    Two batch sizes, 32x apart, because a parameter-sized region cannot
    depend on B and this is where that is visible per region rather than per
    step."""
    B_lo, B_hi = 4, 128
    P = next((r["params"] for k, r in bench.items() if k[1] == canon(arm)[0]),
             None)

    def row(tag, region, touches, label=None):
        t_lo = regions.get((tag, arm, B_lo, region))
        t_hi = regions.get((tag, arm, B_hi, region))
        gb = lambda ms: (None if not (ms and P) else
                         touches * P * BYTES_PER_ELEM / (ms * 1e-3) / 1e9)
        return [r"\texttt{%s}" % tex_escape(label or region), str(touches),
                fmt(t_lo, 4), fmt(gb(t_lo), 1), fmt(t_hi, 4), fmt(gb(t_hi), 1)]

    def total(tag, regs):
        got = [[regions.get((tag, arm, B, r)) for r in regs] for B in (B_lo, B_hi)]
        sums = [None if any(v is None for v in g) else sum(g) for g in got]
        n = sum(regs.values())
        gb = lambda ms: None if not (ms and P) else \
            n * P * BYTES_PER_ELEM / (ms * 1e-3) / 1e9
        return ["all $O(P)$ work", str(n),
                fmt(sums[0], 4), fmt(gb(sums[0]), 1),
                fmt(sums[1], 4), fmt(gb(sums[1]), 1)]

    pre = TOUCHES.get((tag_before, arm), {})
    post = TOUCHES.get((tag_after, arm), {})
    body = [row(tag_before, r, n) for r, n in pre.items()]
    if pre:
        body.append(total(tag_before, pre))
    body += [row(tag_after, r, n) for r, n in post.items()]
    return table(
        ["Region", "Touches", "ms at $B=%d$" % B_lo, "GB/s",
         "ms at $B=%d$" % B_hi, "GB/s"],
        "lrrrrr", body,
        "source: %s (arm %s, %s then %s)"
        % (srcs, arm, tag_before, tag_after))


def amortization_table(bench, tag, srcs):
    """The measured curve: what fraction of a step the posterior owns, as B
    grows with T held. One row per batch size, one column pair per arm.

    The posterior column is DIRECTLY timed -- the Bayesian work is in its own
    kernel launches and each is bracketed by CUDA events -- so no column here
    is a difference against a deterministic baseline that also differs
    elsewhere. Ens is the control: it samples nothing, so its ms/step column
    is what the same network costs with no posterior at all.

    The percentage is ALL parameter-sized work over the step, not only the
    Bayesian part. That is deliberate. Fusing folded the plain-AdamW and the
    memset into the same kernel as the posterior update, so a Bayesian-only
    column stops meaning the same thing before and after, while total O(P)
    work means the same thing in both. It is therefore an upper bound on the
    posterior\'s own share -- and Ens\'s column is exactly what that O(P)
    floor costs when there is no posterior at all."""
    Bs = sorted({k[2] for k in bench if k[0] == tag})
    body = []
    for B in Bs:
        rows = {key: bench.get((tag, key, B)) for key in ARMS}
        got = next((r for r in rows.values() if r), None)
        cells = [str(B), str(got["tokens"]) if got else NA]
        for key in ARMS:
            r = rows.get(key)
            cells += [fmt(r["total"], 2) if r else NA,
                      fmt(100.0 * r["paramwork"] / r["total"], 1) if r else NA]
        body.append(cells)
    header = ["$B$", "Tokens/step"]
    for key in ARMS:
        header += ["%s ms" % key, r"%s $O(P)$\%%" % key]
    return table(header, "rr" + "rr" * len(ARMS), body,
                 "source: %s (variant %s)" % (srcs, tag))


def fusion_table(bench, before, after, srcs):
    """What fusing the parameter-sized passes bought, per arm and per B.

    Both the O(P) columns and the share are TOTAL parameter-sized work, not
    the Bayesian subset: the fused kernel absorbed the plain-AdamW pass and
    the memset, so only the total is comparable across the two variants. See
    amortization_table.

    A row exists only where BOTH variants measured that (arm, B) -- a speedup
    against a point that was never measured is not a measurement.

    The last column is the share after over the share before, at that B. It
    answers a different question from the ratio of times: whether fusing moved
    the amortization curve down or bent it."""
    keys = sorted({(k[1], k[2]) for k in bench if k[0] in (before, after)},
                  key=lambda ab: (ARM_ORDER.get(ab[0], len(ARM_ORDER)), ab[1]))
    body = []
    for arm, B in keys:
        pre, post = bench.get((before, arm, B)), bench.get((after, arm, B))
        if not (pre and post):
            continue
        body.append([
            tex_escape(canon(pre["arm"])[1]), str(B),
            fmt(pre["paramwork"], 2), fmt(post["paramwork"], 2),
            fmt(ratio(pre["paramwork"], post["paramwork"]), 2),
            fmt(pre["total"], 2), fmt(post["total"], 2),
            fmt(100.0 * pre["paramwork"] / pre["total"], 1),
            fmt(100.0 * post["paramwork"] / post["total"], 1),
            fmt(ratio(post["paramwork"] / post["total"],
                      pre["paramwork"] / pre["total"]), 2),
        ])
    return table(
        ["Arm", "$B$", "$O(P)$ ms before", "$O(P)$ ms after", "$\\times$",
         "ms/step before", "ms/step after", "$O(P)$ \\% before",
         "$O(P)$ \\% after", "share ratio"],
        "lrrrrrrrrr", body,
        "source: %s (%s vs %s)" % (srcs, before, after))


def crossover_table(bench, before, after, srcs, band=DRIFT_BAND):
    """Where each arm stops being worth folding, as one batch size per arm.

    Folding deletes a fixed number of buffer touches, so what it returns is a
    fixed wall-clock per step, and a fixed quantity amortises exactly as the
    posterior's own cost does. The decision therefore reduces to the batch size
    at which that fixed saving falls inside `band` -- below which it is not
    separable from the drift the control arm measures.

    The denominator is the UNFUSED step, which is what a reader who has not
    written the fused kernel is paying. The ms columns are total O(P) work, not
    the Bayesian subset, for the reason fusion_table gives: after folding, only
    the total means the same thing in both variants.

    B* is a measured grid point and the grid is octaves, so it brackets the
    crossover rather than locating it. The two ends are rendered as bounds
    rather than as a batch size: an arm already inside the band at the smallest
    point measured has no regime where folding pays, and one still outside it at
    the largest point pays everywhere this card can reach."""
    def paired(key):
        return [(B, bench[(before, key, B)], bench[(after, key, B)])
                for (t, a, B) in sorted(bench, key=lambda k: k[2])
                if t == before and a == key and (after, key, B) in bench]

    seen = [{B for B, _, _ in paired(k)} for k in BAYES_ARMS]
    common = set.intersection(*seen) if all(seen) else set()
    B_lo, B_hi = (min(common), max(common)) if common else (None, None)

    # The control's own before/after difference IS the drift, so a band narrower
    # than what it reproduces would be calling measurement noise a result.
    drift = max((abs(pre["paramwork"] - post["paramwork"]) / pre["total"]
                 for _, pre, post in paired("Ens")), default=0.0)
    if drift > band:
        print("make_tables: control arm drifts %.2f%% between the sweeps, wider "
              "than the %.2f%% band tab-crossover calls drift" %
              (100.0 * drift, 100.0 * band), file=sys.stderr)

    body = []
    for key in BAYES_ARMS:
        rows = paired(key)
        if not rows:
            body.append([key] + [NA] * 6)
            continue
        rec = {B: pre["paramwork"] - post["paramwork"] for B, pre, post in rows}
        frac = {B: rec[B] / pre["total"] for B, pre, _ in rows}
        share = {B: pre["share"] for B, pre, _ in rows}
        Bs = [B for B, _, _ in rows]
        inside = [B for B in Bs if frac[B] <= band]
        if not inside:
            star = "$> %d$" % Bs[-1]
        elif inside[0] == Bs[0]:
            star = r"$\leq %d$" % Bs[0]
        else:
            star = str(inside[0])
        pct = lambda v: NA if v is None else fmt(100.0 * v, 2) + r"\%"
        body.append([
            tex_escape(canon(rows[0][1]["arm"])[1]),
            fmt(min(rec.values()), 3), fmt(max(rec.values()), 3),
            pct(frac.get(B_lo)), pct(frac.get(B_hi)), pct(share.get(B_hi)),
            star,
        ])
    return table(
        ["Arm", "Folding returns (ms, min)", "(max)",
         r"\%% of step at $B=%s$" % (B_lo if B_lo else "?"),
         r"at $B=%s$" % (B_hi if B_hi else "?"),
         r"Posterior \%% at $B=%s$" % (B_hi if B_hi else "?"), "$B^\\ast$"],
        "lrrrrrr", body,
        "source: %s (%s vs %s, band %.0f%%)"
        % (srcs, before, after, 100.0 * band))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=os.path.join(root, "runs", "results.csv"))
    ap.add_argument("--logs", default=os.path.join(root, "runs"),
                    help="directory scanned for *.log eval output")
    ap.add_argument("--out", default=os.path.join(here, "generated"))
    # Which bench variant each table speaks for. The amortization curve is
    # stated for the shipped code (`post`); the fusion table is the pair.
    ap.add_argument("--bench-before", default="pre")
    ap.add_argument("--bench-after", default="post")
    args = ap.parse_args()

    csv_path = args.csv if os.path.exists(args.csv) else None
    log_paths = sorted(glob(os.path.join(args.logs, "*.log")))

    # Paths go into a committed file, so they are stated relative to the
    # repo -- whose machine generated a table is not part of its provenance.
    rel = lambda p: os.path.relpath(p, root)
    csv_rows = read_csv_rows(csv_path)
    records = read_eval_records(log_paths)
    bench = read_bench_records(log_paths)
    regions = read_region_records(log_paths)
    srcs = ("%d log file(s) under %s" % (len(log_paths), rel(args.logs))
            if log_paths else "(no logs found)")

    os.makedirs(args.out, exist_ok=True)
    ood, disagreed = ood_table(records, srcs)
    written = {
        "tab-cost.tex": cost_table(csv_rows, rel(csv_path) if csv_path else None),
        "tab-quality.tex": quality_table(records, srcs),
        "tab-uncertainty.tex": uncertainty_table(records, srcs),
        "tab-ood.tex": ood,
        "tab-posterior.tex": posterior_table(bench, args.bench_before, srcs),
        "tab-bandwidth.tex": bandwidth_table(regions, bench, args.bench_before,
                                             args.bench_after, srcs),
        "tab-amortization.tex": amortization_table(bench, args.bench_after, srcs),
        "tab-fusion.tex": fusion_table(bench, args.bench_before,
                                       args.bench_after, srcs),
        "tab-crossover.tex": crossover_table(bench, args.bench_before,
                                             args.bench_after, srcs),
    }
    made = stamp(root)
    actions = {name: write_fragment(os.path.join(args.out, name), text, made)
               for name, text in written.items()}

    missing = sum(t.count(NA) for name, t in written.items()
                  if actions[name] != "kept")
    print("make_tables: %d csv row(s), %d eval record(s) -> %s"
          % (len(csv_rows), len(records), rel(args.out)))
    for name, what in sorted(actions.items()):
        if what == "kept":
            print("make_tables: %s kept -- nothing measured to replace its "
                  "numbers with" % name)
        elif what == "unchanged":
            print("make_tables: %s unchanged" % name)
    print("make_tables: %d placeholder cell(s) still unmeasured" % missing)
    for key in disagreed:
        print("make_tables: %s reports different OOD epistemic per split -- "
              "the stream is supposed to be split-independent; cell left "
              "unmeasured" % key, file=sys.stderr)
    return 0


# One runnable self-check: the parsers must survive the real formats and
# must never invent a value.  python3 make_tables.py --selftest
def selftest():
    log = ("eval: arm bbb   | split val | S=8 | 64 batches x 4096 tokens"
           " = 262144 ID positions (+262144 OOD)\n"
           "summary: NLPD single 1.3996 -> mixture 1.3257 | ECE(15) 0.0806"
           " | identity_err 1.8e-15\n"
           "summary: uncertainty (nats)  ID: total 1.587 = alea 1.517 +"
           " epi 0.0700   OOD: epi 0.1845  (OOD > ID, as required)\n")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "BBB_eval.log")
        with open(p, "w") as fh:
            fh.write(log)
        rs = read_eval_records([p])
        assert len(rs) == 1, rs
        r = rs[0]
        assert (r["arm"], r["split"], r["S"]) == ("bbb", "val", 8), r
        assert r["nlpd_single"] == 1.3996 and r["nlpd_mix"] == 1.3257, r
        assert r["ece"] == 0.0806 and r["ece_bins"] == 15, r
        assert (r["total"], r["alea"], r["epi"], r["epi_ood"]) == \
            (1.587, 1.517, 0.07, 0.1845), r
        # a log with the summary lines but no eval: header still parses
        p2 = os.path.join(d, "Ens5_train.log")
        with open(p2, "w") as fh:
            fh.write(log.split("\n", 1)[1])
        r2 = read_eval_records([p2])[0]
        assert canon(r2["arm"]) == ("Ens", "Ens5"), r2
        assert r2["split"] is None and r2["S"] is None, r2
    # missing data must produce the macro, never a number
    assert fmt(None, 3) == NA and num("nope") is None
    t = quality_table([], "none")
    assert t.count(NA) == len(ARMS) * len(SPLITS) * 4, t
    assert "0.0" not in t, t
    assert canon("LowR8") == ("LowR", "LowR8") and canon("vadam")[1] == "Vadam"

    # both splits, and no fallback between them: an arm measured only on val
    # must leave its test row unmeasured rather than repeat the val number.
    def rec(arm, split, **kw):
        d = {"arm": arm, "split": split, "S": 32, "mtime": 0.0, "src": "t"}
        d.update(kw)
        return d
    only_val = [rec("bbb", "val", nlpd_mix=1.5582, ece=0.0240, epi=0.0107,
                    epi_ood=0.0119, total=1.602, alea=1.591)]
    q = quality_table(only_val, "none")
    assert "1.5582" in q and q.count("1.5582") == 1, q
    assert pick_split(only_val, "BBB", "test") is None
    # ...and the OOD term is stated ONCE per arm, with the ratio per split
    ood, bad = ood_table(only_val, "none")
    assert not bad and ood.count("0.0119") == 1, ood
    assert "1.112" in ood, ood                      # 0.0119 / 0.0107
    assert ood.count(NA) == len(ARMS) * 4 - 3, ood  # BBB has all but test ratio
    # a split-dependent OOD term breaks the premise of stating it once, so
    # the cell must go unmeasured and say so, not average or pick one
    split_dep = only_val + [rec("bbb", "test", epi=0.0103, epi_ood=0.0130)]
    ood2, bad2 = ood_table(split_dep, "none")
    assert bad2 == ["BBB"] and "0.0119" not in ood2 and "0.0130" not in ood2, ood2

    # ---- the bench tables -------------------------------------------------
    # A synthetic sweep where the posterior is FIXED and the step doubles, so
    # the share must halve and the local exponent must come out at exactly 1.
    def brec(tag, arm, B, post, paramwork, total):
        return {"tag": tag, "arm": arm, "B": B, "T": 64, "tokens": 64 * B,
                "params": 1000000, "steps": 10, "warmup": 1, "total": total,
                "posterior": post, "paramwork": paramwork,
                "model": total - paramwork, "share": post / total,
                "p25": 0.0, "p75": 0.0, "temp_lo": 60, "temp_hi": 70,
                "sm_lo": 1, "sm_hi": 2}
    sweep = {}
    for B, total in ((1, 10.0), (2, 20.0)):
        sweep[("pre", "BBB", B)] = brec("pre", "bbb", B, 1.0, 2.0, total)
        sweep[("post", "BBB", B)] = brec("post", "bbb", B, 1.0, 1.0, total)
    assert abs(local_exponent(sweep[("pre", "BBB", 1)],
                             sweep[("pre", "BBB", 2)]) - 1.0) < 1e-12
    pt = posterior_table(sweep, "pre", "none")
    assert "1.000 & 1.000 & 0.0\\%" in pt, pt        # fixed cost, no spread
    assert "10.00\\% & 5.00\\%" in pt, pt            # share halves
    assert "1.000 & 1.000" in pt, pt                 # k = 1 at both ends
    assert pt.count(NA) == 2 * 8, pt                 # the two unmeasured arms
    # the fusion table's share ratio is post-share over pre-share
    ft = fusion_table(sweep, "pre", "post", "none")
    assert "20.0 & 10.0 & 0.50" in ft and "10.0 & 5.0 & 0.50" in ft, ft

    # the crossover: folding returns a FIXED ms, so its share of the step falls
    # as the step grows, and B* is the first measured point at which it lands
    # inside the drift band. One arm per rendering -- a crossover inside the
    # sweep, one at or below its smallest point, one past its largest.
    xs = {}
    for B, total in ((1, 10.0), (2, 20.0), (4, 100.0)):
        for arm, pre_pw, post_pw in (("bbb", 2.0, 1.0), ("vadam", 2.0, 2.0),
                                     ("lowr", 12.0, 2.0), ("ens", 1.0, 1.0)):
            key = canon(arm)[0]
            xs[("pre", key, B)] = brec("pre", arm, B, 1.5, pre_pw, total)
            xs[("post", key, B)] = brec("post", arm, B, 1.5, post_pw, total)
    ct = crossover_table(xs, "pre", "post", "none")
    assert NA not in ct, ct
    assert r"BBB & 1.000 & 1.000 & 10.00\% & 1.00\% & 1.50\% & 4" in ct, ct
    assert r"Vadam & 0.000 & 0.000 & 0.00\% & 0.00\% & 1.50\% & $\leq 1$" in ct, ct
    assert r"LowR & 10.000 & 10.000 & 100.00\% & 10.00\% & 1.50\% & $> 4$" in ct, ct
    # the band is what B* means: widen it and the same measurement crosses
    # earlier, because less of the saving is separable from drift.
    assert r"1.50\% & 2 " in crossover_table(xs, "pre", "post", "none", 0.06)
    # ...and a control arm that does not reproduce itself inside the band says
    # so, because then the band is not the floor the table claims it is.
    import contextlib, io
    drifted = dict(xs)
    drifted[("post", "Ens", 1)] = brec("post", "ens", 1, 0.0, 2.0, 10.0)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        crossover_table(drifted, "pre", "post", "none")
    assert "control arm drifts 10.00%" in err.getvalue(), err.getvalue()
    assert crossover_table({}, "pre", "post", "none").count(NA) \
        == len(BAYES_ARMS) * 6
    # bandwidth: touches x P x 4 bytes over the measured time, and a region
    # with no measurement stays unmeasured rather than becoming a zero.
    TOUCHES[("pre", "bbb")] = {"r0": 2}
    TOUCHES[("post", "bbb")] = {"r1": 1}
    try:
        regions = {("pre", "bbb", 4, "r0"): 1.0, ("pre", "bbb", 128, "r0"): 2.0}
        bw = bandwidth_table(regions, sweep, "pre", "post", "none")
        assert "1.0000 & 8.0 & 2.0000 & 4.0" in bw, bw   # 2*1e6*4B / 1ms
        assert bw.count(NA) == 4, bw                     # r1 measured nowhere
    finally:
        del TOUCHES[("pre", "bbb")], TOUCHES[("post", "bbb")]

    # a fragment is committed, so a run with nothing measured must leave it
    # alone -- and a rerun that changes nothing but the stamp must not churn.
    measured_tab = quality_table(only_val, "none")
    empty_tab = quality_table([], "none")
    assert has_number(measured_tab) and not has_number(empty_tab)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tab.tex")
        assert write_fragment(p, measured_tab, STAMP + "aaa on day 1") == "wrote"
        assert write_fragment(p, empty_tab, STAMP + "bbb on day 2") == "kept"
        with open(p) as fh:
            assert "1.5582" in fh.read(), "placeholders clobbered a measurement"
        assert write_fragment(p, measured_tab, STAMP + "ccc on day 3") == "unchanged"
        with open(p) as fh:
            assert "aaa on day 1" in fh.read(), "stamp churned on equal numbers"
        assert write_fragment(p, measured_tab.replace("1.5582", "1.5583"),
                              STAMP + "ccc on day 3") == "wrote"
        # ...and a fragment with no cells of its own says so explicitly
        assert write_fragment(p, "% x\n", STAMP + "d", measured=True) == "wrote"
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
