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

Outputs: paper/tables/generated/tab-{cost,quality,uncertainty,ood}.tex, each
a bare `tabular` the document wraps in a float. Those files are COMMITTED --
they are measurements, and CI has no runs/ to remake them from -- so a run
with nothing measured behind it leaves them alone instead of overwriting a
real number with a placeholder. Each carries a `% generated:` stamp saying
which commit and day its numbers came from; that is how a stale one is
spotted. See paper/README.md.

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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=os.path.join(root, "runs", "results.csv"))
    ap.add_argument("--logs", default=os.path.join(root, "runs"),
                    help="directory scanned for *.log eval output")
    ap.add_argument("--out", default=os.path.join(here, "generated"))
    args = ap.parse_args()

    csv_path = args.csv if os.path.exists(args.csv) else None
    log_paths = sorted(glob(os.path.join(args.logs, "*.log")))

    # Paths go into a committed file, so they are stated relative to the
    # repo -- whose machine generated a table is not part of its provenance.
    rel = lambda p: os.path.relpath(p, root)
    csv_rows = read_csv_rows(csv_path)
    records = read_eval_records(log_paths)
    srcs = ("%d log file(s) under %s" % (len(log_paths), rel(args.logs))
            if log_paths else "(no logs found)")

    os.makedirs(args.out, exist_ok=True)
    ood, disagreed = ood_table(records, srcs)
    written = {
        "tab-cost.tex": cost_table(csv_rows, rel(csv_path) if csv_path else None),
        "tab-quality.tex": quality_table(records, srcs),
        "tab-uncertainty.tex": uncertainty_table(records, srcs),
        "tab-ood.tex": ood,
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
