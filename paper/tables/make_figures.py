#!/usr/bin/env python3
"""Generate the paper's figures from measured output. Nothing else.

    python3 paper/tables/make_figures.py       # from the repo root
    make -C paper figures                      # same thing

Sibling of make_tables.py, under the same one rule: a number reaches the
PDF only if it was measured. It reuses make_tables.py's log parsers rather
than owning a second copy of the stdout format.

Inputs (all optional; missing data becomes a placeholder, never a value):

  runs/sweep/*.log   eval stdout from an S sweep, one file per (arm, S).
  runs/*.log         eval stdout for the ensemble's structural S=5 point.

Output: paper/tables/generated/fig-s-convergence.{pdf,tex}. The .tex is what
the section \\input s -- it is either an \\includegraphics of the .pdf, or the
\\todo macro when there was nothing to plot. That indirection is what lets
the paper build on a checkout with no runs/ (CI) without a missing-file
error and without a hand-drawn stand-in.

Both are COMMITTED, like the tables, so that checkout gets the real figure:
with no sweep data this script leaves whatever is on disk alone rather than
replacing a plot with the placeholder. It still deletes a figure whose data
disappeared while runs/ was there to look at.

THE S=1 INVARIANT: one sample disagrees with nothing, so epistemic at S=1
must be exactly zero. It is the sanity check of the whole plot. If it is not
exactly zero this script refuses to plot and exits nonzero, rather than
drawing a curve whose foot is wrong.

The marked S is the suite's evaluation convention and is DERIVED from these
logs, never typed: see converged_S(). An arm still climbing at the end of the
range leaves it unmarked instead of settling it on the arms that converged.
"""

import argparse
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (path must be set first)

# Series colours: slots 1-4 of the validated categorical palette, in its
# order. A line chart is scored on the adjacent pairlist, which those four
# pass: worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6,
# light surface. Aqua and yellow sit under 3:1 on white, so every series is
# also direct-labelled.
COLORS = {"BBB": "#2a78d6", "LowR": "#eb6834", "Ens": "#1baf7a",
          "Vadam": "#eda100"}
MARKERS = {"BBB": "o", "LowR": "s", "Ens": "D", "Vadam": "^"}
# One display name per arm, used by BOTH the direct label and the legend --
# two names for one series reads as two series.
NAMES = {"BBB": "BBB", "LowR": "LowR8", "Ens": "Ens5", "Vadam": "Vadam"}
INK, MUTED, RULE = "#3a3a38", "#52514e", "#9a9a95"
SWEEPABLE = ["BBB", "Vadam", "LowR"]   # arms whose S is a free parameter
PINNED = "Ens"                         # S is the number of trained members

# uq_report prints epistemic with four decimals, so 1e-4 is the finest
# difference a log can express and there is no tolerance to choose here: an
# arm has converged at the smallest S from which it and every larger S it was
# measured at are indistinguishable, at the precision the harness prints,
# from the largest. Derived from the format string, not tuned.
EPI_QUANTUM = 1e-4


def sweep_points(records):
    """{arm: [(S, epi), ...]} sorted by S, one point per S. A repeated S
    keeps the most recent log, matching make_tables.py's tie-break."""
    best = {}
    for r in records:
        key = mt.canon(r["arm"])[0]
        if r.get("S") is None or r.get("epi") is None:
            continue
        k = (key, r["S"])
        if k not in best or r["mtime"] >= best[k]["mtime"]:
            best[k] = r
    out = {}
    for (key, S), r in best.items():
        out.setdefault(key, []).append((S, r["epi"], r))
    for key in out:
        out[key].sort()
    return out


def converged_S(points):
    """The suite's S, read off the sweep rather than typed: the smallest
    measured S at which EVERY sweepable arm has converged. None when some arm
    has not, or was not swept -- a convention no arm's data supports is not
    one to draw. The slowest arm sets it; reporting below it would state that
    arm's epistemic term under its own converged value."""
    per_arm = []
    for key in SWEEPABLE:
        pts = points.get(key)
        if not pts or len(pts) < 2:
            return None
        # Compared as printed digits, not as floats: 0.0108 and 0.0107 differ
        # by one digit and by 1.0000000000000005e-4, and a float comparison
        # against the quantum would call that a difference.
        # The largest S measured is never itself the answer, either: it agrees
        # with itself whatever the curve is doing, so convergence has to be
        # confirmed by at least one further doubling.
        digits = lambda e: int(round(e / EPI_QUANTUM))
        final = digits(pts[-1][1])
        flat = [pts[i][0] for i in range(len(pts) - 1)
                if all(abs(digits(e) - final) <= 1 for _, e, _ in pts[i:])]
        if not flat:
            return None
        per_arm.append(flat[0])
    return max(per_arm)


def check_s1(points):
    """Returns a list of violations of the S=1 invariant."""
    bad = []
    for key, pts in sorted(points.items()):
        for S, epi, r in pts:
            if S == 1 and epi != 0.0:
                bad.append("%s S=1 epistemic %r (expected exactly 0.0) in %s"
                           % (key, epi, r["src"]))
    return bad


def plot(points, ens, out_pdf, split, mark):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Authored near a one-column text width (6.5in at the scaffold's 1in
    # margins) so \includegraphics[width=\linewidth] barely rescales it and
    # the tick labels land at roughly their nominal point size in the PDF.
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    drawn = []
    for key in SWEEPABLE:
        pts = points.get(key)
        if not pts:
            continue
        xs = [S for S, _, _ in pts]
        ys = [e for _, e, _ in pts]
        ax.plot(xs, ys, marker=MARKERS[key], color=COLORS[key], linewidth=2,
                markersize=5, markeredgecolor="white", markeredgewidth=0.8,
                zorder=3, clip_on=False)
        ax.annotate(NAMES[key], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(7, 0), color=INK, fontsize=9, va="center",
                    annotation_clip=False)
        drawn.append(key)
    if ens:
        S, epi = ens
        ax.plot([S], [epi], marker=MARKERS[PINNED], color=COLORS[PINNED],
                markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                linestyle="none", zorder=4, clip_on=False)
        ax.annotate(NAMES[PINNED], (S, epi), textcoords="offset points",
                    xytext=(9, 0), color=INK, fontsize=9, va="center",
                    annotation_clip=False)
        drawn.append(PINNED)

    # The convention, drawn where the sweep puts it. It is behind the curves
    # and labelled in the empty band below them, the one region no series and
    # no legend occupies.
    if mark:
        ax.axvline(mark, color=RULE, linewidth=1, linestyle=(0, (4, 3)),
                   zorder=1, ymax=0.7)   # stops short of the legend
        ax.annotate("suite $S = %d$: every sweepable arm converged" % mark,
                    (mark, 0), textcoords="offset points", xytext=(-6, 6),
                    color=MUTED, fontsize=8, ha="right", va="bottom")

    all_S = sorted({S for pts in points.values() for S, _, _ in pts})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_S)
    ax.set_xticklabels([str(s) for s in all_S])
    ax.set_xlim(min(all_S) * 0.92, max(all_S) * 1.42)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$S$ (posterior samples per position)")
    ax.set_ylabel("epistemic uncertainty (nats)")
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK, labelsize=9)
    # Identity is never colour-alone: every series is direct-labelled above
    # AND legended here, under one name each. Upper right is the one region
    # no mark occupies -- the curves sit low, the pinned point sits left.
    ax.legend(handles=[
        matplotlib.lines.Line2D(
            [], [], color=COLORS[k], marker=MARKERS[k], markersize=5,
            markeredgecolor="white", linestyle="none" if k == PINNED else "-",
            linewidth=2, label=NAMES[k] + (" ($S$ fixed by the arm at %d)" % ens[0]
                                           if k == PINNED else ""))
        for k in drawn],
        frameon=False, fontsize=8, loc="upper right", labelcolor=INK,
        handletextpad=0.6, borderaxespad=0.2)
    ax.set_title("split: %s" % split, fontsize=9, color=MUTED, loc="left")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_pdf, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figure_tex(have_pdf, note):
    head = ["% GENERATED by paper/tables/make_figures.py -- do not edit.",
            "%% %s" % note]
    if have_pdf:
        body = [r"\includegraphics[width=\linewidth]"
                r"{\generated/fig-s-convergence.pdf}"]
    else:
        body = [r"\todo{S-convergence figure: no measured sweep data "
                r"under runs/sweep/ -- see paper/README.md}"]
    return "\n".join(head + body) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", default=os.path.join(root, "runs", "sweep"),
                    help="directory scanned for *.log S-sweep output")
    ap.add_argument("--logs", default=os.path.join(root, "runs"),
                    help="directory scanned for the pinned arm's *.log")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=os.path.join(here, "generated"))
    args = ap.parse_args()

    sweep_logs = sorted(glob(os.path.join(args.sweep, "*.log")))
    points = sweep_points([r for r in mt.read_eval_records(sweep_logs)
                           if r.get("split") == args.split
                           and mt.canon(r["arm"])[0] in SWEEPABLE])

    bad = check_s1(points)
    if bad:
        for b in bad:
            print("make_figures: FAIL " + b, file=sys.stderr)
        print("make_figures: refusing to plot -- the S=1 invariant is the "
              "sanity check of this figure", file=sys.stderr)
        return 1

    e = mt.pick(mt.read_eval_records(sorted(glob(
        os.path.join(args.logs, "*.log")))), PINNED, args.split)
    ens = (e["S"], e["epi"]) if e and e.get("S") and e.get("epi") is not None \
        else None

    os.makedirs(args.out, exist_ok=True)
    out_pdf = os.path.join(args.out, "fig-s-convergence.pdf")
    out_tex = os.path.join(args.out, "fig-s-convergence.tex")
    rel = lambda p: os.path.relpath(p, root)
    have = bool(points)

    # No sweep logs AT ALL is a checkout with no runs/ (CI): the committed
    # figure is then the only measurement in existence, so keep it. Logs that
    # stopped yielding points is data that broke while runs/ was right there,
    # and a plot must never outlive its data -- that path still deletes.
    if not have and not sweep_logs and os.path.exists(out_tex):
        print("make_figures: kept the committed figure -- no sweep logs "
              "under %s to remake it from" % rel(args.sweep))
        return 0

    mark = converged_S(points)
    note = ("source: %d sweep log(s) under %s, split %s%s; suite S %s"
            % (len(sweep_logs), rel(args.sweep), args.split,
               "" if ens else "; pinned-arm point NOT measured",
               mark if mark else "unmarked -- some sweepable arm has not "
               "converged over the range measured"))
    if have:
        plot(points, ens, out_pdf, args.split, mark)
    elif os.path.exists(out_pdf):
        os.remove(out_pdf)   # never let a stale figure outlive its data
    mt.write_fragment(out_tex, figure_tex(have, note), mt.stamp(root),
                      measured=have)

    for key in SWEEPABLE:
        pts = points.get(key, [])
        print("make_figures: %-5s %d point(s): %s" % (
            key, len(pts), " ".join("S=%d:%.4f" % (S, e) for S, e, _ in pts)))
    print("make_figures: %-5s %s" % (
        PINNED, "S=%d:%.4f" % ens if ens else "not measured -- omitted"))
    print("make_figures: suite S %s" % (mark if mark else "unmarked"))
    print("make_figures: %s" % ("wrote " + out_pdf if have
                                else "no data -> placeholder only"))
    return 0


# One runnable self-check: the S=1 gate must bite, and no data must produce
# the macro rather than a plot.  python3 make_figures.py --selftest
def selftest():
    def rec(arm, S, epi):
        return {"arm": arm, "split": "val", "S": S, "epi": epi,
                "mtime": 0.0, "src": "test"}

    pts = sweep_points([rec("bbb", 1, 0.0), rec("bbb", 2, 0.0052),
                        rec("lowr", 1, 0.0)])
    assert check_s1(pts) == [], pts
    assert pts["BBB"] == [(1, 0.0, pts["BBB"][0][2]),
                          (2, 0.0052, pts["BBB"][1][2])], pts
    # a non-zero S=1 must be caught, not plotted
    assert len(check_s1(sweep_points([rec("bbb", 1, 1e-9)]))) == 1
    # ...including a value that only *prints* as 0.0000
    assert len(check_s1(sweep_points([rec("bbb", 1, 0.00004)]))) == 1
    # the suite's S is the SLOWEST sweepable arm's convergence point, and an
    # arm still climbing at the end of the range leaves it unmarked rather
    # than settling it on the arms that did converge.
    def series(arm, ys):
        return [rec(arm, 2 ** i, y) for i, y in enumerate(ys)]

    fast = series("bbb", [0.0, 0.0090, 0.0107, 0.0107])          # flat from 4
    vad = series("vadam", [0.0, 0.0300, 0.0381, 0.0382])         # flat from 4
    slow = series("lowr", [0.0, 0.0100, 0.0150, 0.0171, 0.0171])  # flat from 8
    assert converged_S(sweep_points(fast + vad + slow)) == 8
    assert converged_S(sweep_points(fast + vad)) is None   # LowR not swept
    climbing = series("lowr", [0.0, 0.0100, 0.0150, 0.0200])
    assert converged_S(sweep_points(fast + vad + climbing)) is None
    # one printed digit of movement is not movement -- it is all a log of
    # four decimals can express -- but two is.
    wiggle = series("lowr", [0.0, 0.0100, 0.0150, 0.0170, 0.0171])
    assert converged_S(sweep_points(fast + vad + wiggle)) == 8
    two = series("lowr", [0.0, 0.0100, 0.0150, 0.0169, 0.0171])
    assert converged_S(sweep_points(fast + vad + two)) is None

    # no data -> the macro, never a figure
    t = figure_tex(False, "none")
    assert mt.NA not in t and r"\todo" in t and "includegraphics" not in t, t
    assert r"\includegraphics" in figure_tex(True, "none")
    # the figure fragment carries no cells, so it must declare measured=
    # explicitly: a plot on disk survives a run that had nothing to plot.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fig.tex")
        assert mt.write_fragment(p, figure_tex(True, "a"), mt.STAMP + "x",
                                 measured=True) == "wrote"
        assert mt.write_fragment(p, figure_tex(False, "b"), mt.STAMP + "y",
                                 measured=False) == "kept"
        with open(p) as fh:
            assert r"\includegraphics" in fh.read(), "placeholder ate the figure"
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
