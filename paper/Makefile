.PHONY: paper clean

paper: main.tex sections/*.tex references.bib
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.synctex.gz *.fdb_latexmk *.fls
	rm -f sections/*.aux
