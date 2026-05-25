PAPER = paper

all: $(PAPER).pdf

# Build figures first
figures:
	cd figures && python fig1_substrate_flat.py
	cd figures && python fig2_no_forgetting.py
	cd figures && python fig3_separation.py
	cd figures && python fig4_vision_results.py

# Full LaTeX build (pdflatex + bibtex + pdflatex + pdflatex)
$(PAPER).pdf: $(PAPER).tex references.bib figures
	pdflatex $(PAPER).tex
	bibtex $(PAPER)
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex

clean:
	rm -f $(PAPER).pdf $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg $(PAPER).out

distclean: clean
	rm -f figures/*.pdf figures/*.png

.PHONY: all figures clean distclean
