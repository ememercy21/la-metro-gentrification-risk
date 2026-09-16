# Paper source

The submitted scholarly article is available at
[`reports/scholarly_article.pdf`](../reports/scholarly_article.pdf).

To rebuild it from source, install a LaTeX distribution with `latexmk`, then run:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Intermediate LaTeX files are ignored by Git.
