# flatmem paper

LaTeX source for the flatmem research paper.

## Build

```bash
# 1. Generate figures (matplotlib, numpy)
cd figures
python fig1_substrate_flat.py
python fig2_no_forgetting.py
python fig3_separation.py
python fig4_vision_results.py
cd ..

# 2. Compile LaTeX (needs pdflatex + bibtex)
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or with the Makefile:

```bash
make           # build everything
make figures   # regenerate figures only
make clean     # remove build artifacts
```

## Files

```
paper/
├── paper.tex                  main LaTeX source (~12 pages compiled)
├── references.bib             BibTeX
├── Makefile
├── README.md                  this file
└── figures/
    ├── fig1_substrate_flat.py     substrate vs dict footprint over data scale
    ├── fig2_no_forgetting.py      original-5 retention through distractor floods
    ├── fig3_separation.py         mean-removal rank sweep + scale stability
    ├── fig4_vision_results.py     vision channel MNIST accuracy
    └── fig{1,2,3,4}_*.{pdf,png}   generated figures (PDF for LaTeX, PNG for web)
```

## Data sources

Figure 1 (substrate flat) uses vocabulary counts from Pack 125 (100K TinyStories).

Figure 2 (no-forgetting) uses retention measurements from Pack 129.
Transformer baseline is illustrative; exact baseline depends on
architecture and training regime.

Figure 3 (separation) uses Pack 115 (15K stories, mean-removal rank sweep)
and Pack 125 (separation at 25K/50K/75K/100K with r=1).

Figure 4 (vision) uses Pack 127/128 (sklearn 8x8 digits, 1257 train / 540
test, train acc 100% / test acc 90% / save-load 86%).

All raw experiment logs are available in the main NeuroSeed repository
under `experiments/day57_pack11{4..6}_*.py` and `experiments/day58_pack12{7,8,9}_*.py`.

## Submitting to arXiv

1. Build the PDF locally with `make` to verify.
2. Compress the directory:
   ```bash
   tar czf flatmem.tar.gz paper.tex references.bib figures/*.pdf
   ```
3. Upload to https://arxiv.org/submit.
4. Suggested categories: `cs.NE` (Neural and Evolutionary), `cs.AI`,
   `cs.LG` (cross-listed).
