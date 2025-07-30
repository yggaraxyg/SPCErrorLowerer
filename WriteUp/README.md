README:

The two folders (Report and MainProjectSlides) contain .tex files which can be compiled into the report and slideshow for my main project, under the names Report.tex and slides.tex.

TeX_Live 2025 is required to compile these files properly. Additionally, the packager biber needs to be installed. it can be installed with

sudo apt install texlive-bibtex-extra


In order to compile, one needs to use these commands in this order:

pdflatex [name].tex
biber [name]
pdflatex [name].tex
pdflatex [name].tex
pdflatex [name].tex

where [name] is Report or slides depending on which document one is compiling.

(I am aware that the first and last three lines are exactly the same. This is not an error. Each compilation affects further compilations. All five lines are necessary.)

To open SideProjectSlides, one simply needs a slideshow viewer. If one doesn't already have a slideshow editer/viewer installed, I reccomend LibreOffice Impress.