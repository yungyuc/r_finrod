LATEX = latex
DVIPS = dvips
PS2PDF = ps2pdf

TRUNK = finrod

MAKEFILES = Makefile

default: $(TRUNK)_draft.pdf
	
$(TRUNK)_draft.dvi: $(TRUNK)_draft.tex $(MAKEFILES)
	$(LATEX) $<
	$(LATEX) $<
	$(LATEX) $<

$(TRUNK)_draft.ps: $(TRUNK)_draft.dvi $(MAKEFILES)
	$(DVIPS) $<
	
$(TRUNK)_draft.pdf: $(TRUNK)_draft.ps $(MAKEFILES)
	$(PS2PDF) $<

.PHONY: clean
clean:
	rm -f \
		$(TRUNK)_*.aux \
		$(TRUNK)_*.dvi \
		$(TRUNK)_*.log \
		$(TRUNK)_*.out \
		$(TRUNK)_*.pdf \
		$(TRUNK)_*.ps
