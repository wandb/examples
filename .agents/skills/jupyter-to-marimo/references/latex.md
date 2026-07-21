# Porting LaTeX from Jupyter to marimo

Jupyter uses **MathJax**. marimo uses **KaTeX** (faster, slightly narrower coverage, silent errors).

## Use raw strings

LaTeX lives in Python strings in marimo, so use `r"..."` to preserve backslashes:

```python
mo.md(r"$\frac{1}{2}$")   # correct
mo.md("$\frac{1}{2}$")    # wrong — \f is a form-feed character
```

## Jupyter (MathJax) → marimo (KaTeX)

| Category | Jupyter (MathJax) | marimo (KaTeX) |
| --- | --- | --- |
| Text | `\mbox`, `\bbox` | `\text{}` |
| Text style | `\textsc`, `\textsl` | `\text{}` |
| Environments | `\begin{eqnarray}` | `\begin{align}` |
| | `\begin{multline}` | `\begin{gather}` |
| References | `\label`, `\eqref`, `\ref` | `\tag{}` for manual numbering |
| Arrays | `\cline`, `\multicolumn`, `\hfill`, `\vline` | not supported |
| Macros | `\DeclareMathOperator` | `\operatorname{}` inline |
| | `\newenvironment` | not supported |
| Spacing | `\mspace`, `\setlength`, `\strut`, `\rotatebox` | not supported |
| Conditionals | `\if`, `\else`, `\fi`, `\ifx` | not supported |

**These DO work** in KaTeX (despite outdated claims): `\newcommand`, `\def`, `\hbox`, `\hskip`, `\cal`, `\pmb`, `\begin{equation}`, `\begin{split}`, `\operatorname*`.

## Migration checklist

1. Find-replace `\mbox{` → `\text{`
2. Use raw strings (`r"..."`)
3. Replace `\begin{eqnarray}` → `\begin{align}`
4. Replace `\DeclareMathOperator` → `\operatorname{}`
5. Remove `\label`/`\eqref` → use `\tag{}` if needed
6. Visually verify — KaTeX fails silently

## References

- [KaTeX Support Table](https://katex.org/docs/support_table) — definitive command lookup
- [KaTeX Unsupported Features](https://github.com/KaTeX/KaTeX/wiki/Things-that-KaTeX-does-not-(yet)-support)
