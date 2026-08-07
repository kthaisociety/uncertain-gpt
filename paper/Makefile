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
TABLES   := tables/generated/tab-cost.tex \
            tables/generated/tab-quality.tex \
            tables/generated/tab-uncertainty.tex

.PHONY: all pdf tables check clean distclean
all: pdf

tables:
	$(PYTHON) $(TABLEGEN)

# .PHONY tables would force a rebuild anyway; this rule exists so the PDF
# has a real dependency on the generated files.
$(TABLES): tables
	@:

pdf: $(TABLES)
ifdef LATEXMK
	$(LATEXMK) -pdf -halt-on-error -interaction=nonstopmode $(DOC).tex
else
	pdflatex -halt-on-error -interaction=nonstopmode $(DOC).tex
	-bibtex $(DOC)
	pdflatex -halt-on-error -interaction=nonstopmode $(DOC).tex
	pdflatex -halt-on-error -interaction=nonstopmode $(DOC).tex
endif

# The generator's self-check. No GPU, no LaTeX.
check:
	$(PYTHON) $(TABLEGEN) --selftest

clean:
ifdef LATEXMK
	$(LATEXMK) -C $(DOC).tex
else
	rm -f $(DOC).aux $(DOC).bbl $(DOC).blg $(DOC).log $(DOC).out \
	      $(DOC).toc $(DOC).fdb_latexmk $(DOC).fls $(DOC).pdf
endif

distclean: clean
	rm -rf tables/generated
