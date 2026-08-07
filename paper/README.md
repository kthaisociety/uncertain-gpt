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
removes build artefacts, `distclean` also removes the generated tables.

Nothing in this directory touches the GPU.

The committed [`main.pdf`](main.pdf) is built by CI
(`.github/workflows/paper.yml`), which has no `runs/`, so **every generated
table cell in it is a red `TODO` placeholder**. For real numbers, build
locally with `runs/` present — that PDF is the one to trust.

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

and writes `tables/generated/tab-{cost,quality,uncertainty}.tex`, which the
results sections `\input`. Those files are **gitignored on purpose**: a
committed table could outlive the measurement it came from.

The one rule the generator enforces: a cell is either a number that was
measured or the macro `\NA`, which `preamble.tex` renders as a loud red
`TODO`. It never fills a gap with a plausible value. `runs/` is not in git,
so on a fresh checkout every cell is `\NA` and the document still builds —
that is the intended behaviour, not a failure.

To soften the placeholders for a camera-ready, redefine `\NA` and `\todo`
in `preamble.tex`. Never by editing a generated table.

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
tables/generated/     its output (gitignored, rebuilt by make)
```
