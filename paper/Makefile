# Build the paper.  `make` from inside paper/.
#
# Tables are ALWAYS regenerated before the PDF: a number in the PDF cannot
# be older than the measurement it came from, and cannot be hand-edited.
#
# latexmk is used when present (it runs pdflatex/bibtex to convergence);
# otherwise the classic four-pass fallback, which needs nothing but a
# TeX Live install with pdflatex and bibtex.

DOC      := main
PYTHON   := python3
LATEXMK  := $(shell command -v latexmk 2>/dev/null)
TABLEGEN := tables/make_tables.py
FIGGEN   := tables/make_figures.py
TABLES   := tables/generated/tab-cost.tex \
            tables/generated/tab-quality.tex \
            tables/generated/tab-uncertainty.tex \
            tables/generated/tab-ood.tex \
            tables/generated/tab-posterior.tex \
            tables/generated/tab-bandwidth.tex \
            tables/generated/tab-amortization.tex \
            tables/generated/tab-fusion.tex
FIGURES  := tables/generated/fig-s-convergence.tex

.PHONY: all pdf tables figures check clean distclean
all: pdf

tables:
	$(PYTHON) $(TABLEGEN)

# Needs matplotlib, but only when there is sweep data to plot: with no
# runs/ (a fresh checkout, CI) the generator emits the \todo placeholder
# without importing it, so the paper still builds with nothing installed.
figures:
	$(PYTHON) $(FIGGEN)

# .PHONY tables/figures would force a rebuild anyway; these rules exist so
# the PDF has a real dependency on the generated files.
$(TABLES): tables
	@:
$(FIGURES): figures
	@:

pdf: $(TABLES) $(FIGURES)
ifdef LATEXMK
	$(LATEXMK) -pdf -halt-on-error -interaction=nonstopmode $(DOC).tex
else
	pdflatex -halt-on-error -interaction=nonstopmode $(DOC).tex
	-bibtex $(DOC)
	pdflatex -halt-on-error -interaction=nonstopmode $(DOC).tex
	pdflatex -halt-on-error -interaction=nonstopmode $(DOC).tex
endif

# The generators' self-checks. No GPU, no LaTeX.
check:
	$(PYTHON) $(TABLEGEN) --selftest
	$(PYTHON) $(FIGGEN) --selftest

clean:
ifdef LATEXMK
	$(LATEXMK) -C $(DOC).tex
else
	rm -f $(DOC).aux $(DOC).bbl $(DOC).blg $(DOC).log $(DOC).out \
	      $(DOC).toc $(DOC).fdb_latexmk $(DOC).fls $(DOC).pdf
endif

# tables/generated holds committed measurements now, so this throws away
# real numbers: `git checkout -- tables/generated` puts them back, and
# regenerating needs runs/ (see README.md).
distclean: clean
	rm -rf tables/generated
