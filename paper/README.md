# paper/

LaTeX scaffolding for the uncertain-gpt paper. **Scaffolding only** — every
section file holds guiding comments and no prose, and every number reaches
the PDF through `tables/`, never by hand.

## Build

```
make -C paper          # regenerate tables, then build paper/main.pdf
```

Needs `pdflatex` and `bibtex` (TeX Live). `latexmk` is used automatically
if it is installed; the Makefile falls back to the classic
pdflatex/bibtex/pdflatex/pdflatex passes if it is not. `make -C paper clean`
removes build artefacts; `distclean` also removes the generated tables and
figure, which are committed — `git checkout -- paper/tables/generated`
puts them back, and regenerating them needs `runs/`.

Nothing in this directory touches the GPU.

The committed [`main.pdf`](main.pdf) is built by CI
(`.github/workflows/paper.yml`), which has no `runs/`. It carries real
numbers anyway, because the generated tables and figure are committed too —
see [Committed fragments](#committed-fragments) for what that does and does
not guarantee.

## Regenerating the tables

```
make -C paper tables                  # what the build already does for you
python3 paper/tables/make_tables.py   # same, run directly
python3 paper/tables/make_tables.py --selftest   # parser self-check
```

`make_tables.py` reads

- `runs/results.csv` — appended per run by `scorecard.c`; cost columns.
- `runs/*.log` — trainer/eval stdout; the uncertainty columns come from the
  two `summary:` lines `uq_report()` prints in `gpt2_kernels.cuh`, and the
  `eval:` line above them supplies the split and *S*.

and writes `tables/generated/tab-{cost,quality,uncertainty,ood}.tex`, which
the results sections `\input`. Those files are **committed**; see below.

The quality and uncertainty tables carry **one row per (arm, split)** and
prefer neither split — which split the paper standardises on is still open,
and a table that silently picked would settle it by accident.
`tab-ood.tex` exists because the off-distribution epistemic term is **not**
a per-split measurement: the OOD stream is a fixed synthetic token sequence
(a seed, not a split), so it is stated once per arm, with the OOD/ID ratio
given per split because only the denominator moves. If the two splits ever
report *different* OOD values that cell goes unmeasured and the generator
says so on stderr — the premise of stating it once would have broken.

The one rule the generator enforces: a cell is either a number that was
measured or the macro `\NA`, which `preamble.tex` renders as a loud red
`TODO`. It never fills a gap with a plausible value.

To soften the placeholders for a camera-ready, redefine `\NA` and `\todo`
in `preamble.tex`. Never by editing a generated table.

## Committed fragments

`runs/` is not in git — the checkpoints are hundreds of megabytes and the
logs come off a GPU. If the generated fragments were not in git either,
every clone and every CI build would produce a PDF whose every cell reads
`TODO`, which is what used to happen. So `tables/generated/*.tex` and
`fig-s-convergence.pdf` are committed, and nothing else in that directory
is.

That does not weaken the one rule. The numbers in a committed fragment
*were* measured; committing them only stops the measurement being thrown
away by the next build on a machine that has no `runs/`. Three things keep
it honest:

- **A run with nothing measured never overwrites a fragment.** With no
  `runs/`, `make -C paper` regenerates nothing and says so
  (`tab-cost.tex kept -- nothing measured to replace its numbers with`).
  With `runs/` present it overwrites normally, so a regeneration is still
  the only way a number changes.
- **Each fragment records where it came from**, as the second line:

      % generated: 7844e83 on 2026-08-07

  the commit (`git describe --always --dirty`) and the UTC day of the run
  that produced it. **That is the staleness signal.** A fragment is suspect
  when its commit is behind anything that could have changed the
  measurement — the kernels, the eval harness, the arm's own `.cu`, the
  generator — which `git log <that commit>..HEAD -- gpt2_kernels.cuh
  train_*.cu paper/tables/` answers directly. `-dirty` in the stamp means
  it was generated from a working tree that did not match any commit, so
  the commit alone does not identify the code that produced it.
- **The stamp does not churn.** Rebuilding with the same measurements
  leaves the file byte-identical (the date only moves when a number does),
  so `git status` after a build is a real signal.

The generators are the only things that write these files. Editing one by
hand is the one way to get a number into the PDF that was never measured.

## Regenerating the figures

```
make -C paper figures                  # what the build already does for you
python3 paper/tables/make_figures.py --selftest   # generator self-check
```

`make_figures.py` is a sibling of `make_tables.py` under the same rule, and
reuses its log parsers rather than owning a second copy of the stdout
format. It reads `runs/sweep/*.log` (an S sweep, one eval log per arm and
*S*) plus `runs/*.log` for the ensemble's structural *S*, and writes
`tables/generated/fig-s-convergence.{pdf,tex}`.

The section `\input`s the **`.tex`**, never the PDF directly: that file
holds either an `\includegraphics` or a red `\todo`, so a checkout with
neither measurements nor a committed figure still builds rather than
failing on a missing image. Plotting needs `matplotlib`; the placeholder
path does not import it, and neither does keeping the committed figure, so
CI needs nothing installed.

Two invariants the generator enforces, beyond the shared one:

- **Epistemic at *S*=1 must be exactly zero** — one sample disagrees with
  nothing. If it is not, the generator refuses to plot and exits nonzero.
- A figure is **deleted, not kept**, when its data disappears *while
  `runs/sweep/` is there to look at* — a plot must not outlive logs that
  stopped producing it. No sweep logs at all is the other case: that is a
  checkout without `runs/`, where the committed figure is the only
  measurement in existence, so it is kept untouched.

## Swapping in a venue template

The preamble is deliberately class-agnostic, so the swap is:

1. Drop the venue's `.sty`/`.cls` next to `main.tex` (both are whitelisted
   by `.gitignore`).
2. Change **one line** — `\documentclass[11pt]{article}` at the top of
   `main.tex` — to whatever the venue wants.
3. In `preamble.tex`, delete the `\usepackage` lines the venue style
   already loads. In practice that is `geometry` (venue styles set their
   own page geometry) and often `natbib` and `hyperref`.
4. Set `\bibliographystyle` in `main.tex` to the venue's `.bst`.

Nothing else in `main.tex` is class-specific, and no section file or
generated table refers to the document class at all.

## Layout

```
main.tex              documentclass + \input of everything; the swap point
preamble.tex          packages and the \todo / \NA placeholder macros
sections/*.tex        one file per section, guiding comments only
references.bib        verified entries; see the header before adding one
tables/make_tables.py the only thing allowed to put a number in the PDF
tables/generated/     its output: the .tex fragments and the figure PDF are
                      committed and stamped, everything else there is not
```
