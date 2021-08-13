pip install pandoc-eqnos




#$ pandoc -s em.md -o em.pdf --pdf-engine=pdflatex
# pandoc -s em.md --citeproc --csl=chicago-syllabus.csl --filter pandoc-eqnos -o em.pdf