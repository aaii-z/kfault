# Paper figures

Regenerate with:

```bash
uv run modal run export.py --level 2 --with-code   # refresh data/ from the volume
uv run python figures.py                           # rebuild every figure
```

Both PDF (vector, for LaTeX `\includegraphics`) and PNG are written. **No number
is typed into `figures.py`** -- every value is computed from `data/attempts.csv`
and `data/chains.csv`, so re-running after a new experiment updates the figures
with no edits.

| figure | what it argues | use it for |
|--------|----------------|------------|
| `fig1_relocation` | false-pass rate rises under repair in all three models (14→20, 16→19, 7→10%) | **the headline claim** |
| `fig2_repair_outcomes` | rescued vs hidden vs still-failing, per model | the mechanism behind fig 1 |
| `fig3_taxonomy` | full category distribution, one-shot vs repaired, both arms | the taxonomy / RQ1 |
| `fig4_probes` | which stress probe catches the silent failures | what the silent bugs *are* |
| `fig5_speedup` | passing kernels are barely faster than PyTorch | the "so what" on performance |

## Reading notes

**fig1** is a slope chart because the claim is about *direction of change*, not
about two magnitudes. Every line goes up; that is the finding.

**fig4** is the strongest single figure for the methodology section: 79 of 86
silent failures are caught by the non-contiguous-input probe alone. That is an
argument for the stress battery being cheap to adopt -- one probe does most of
the work.

**fig5** excludes one extreme outlier: GLM-5 on `80_Gemm_Max_Subtract_GELU`,
2670x on L4 and 6514x on T4. Inspected -- the kernel is **correct**. It proves
in a comment that `max(dim=1)` then `mean(dim=1)` then subtract is identically
zero, so `GELU(0) = 0`, and returns a zero tensor without doing the GEMM.

That is a degenerate benchmark task, not a bad kernel: problem 80's output does
not depend on its input, so any speedup measured on it is meaningless. Medians
are used throughout for that reason, and the task should be excluded from
performance claims. Worth reporting in its own right -- and note the `direct`
arm never found the identity, failing all four turns on both GPUs, while
`think` found it at turn 0.

## Colour

Categorical slots from the dataviz reference palette, checked with its
validator in light mode:

- 5-category stack (fig 3): `#2a78d6,#4a3aa7,#e34948,#eda100,#008300` --
  adjacent CVD dE 13.0, normal-vision 16.3, ALL PASS
- 3-model lines (fig 1, 5): `#2a78d6,#eb6834,#1baf7a` all-pairs -- CVD 9.2,
  normal-vision 24.0, ALL PASS

The category ramp encodes severity, and the reader should be able to see it
without the legend:

| category | hue | means |
|---|---|---|
| syntactic | blue | does not build |
| runtime | violet | crashes |
| functional | red | wrong answer |
| silent | amber | secretly wrong -- passes the standard check |
| pass | green | correct |

`functional` must not be a green. An earlier version used aqua and it read as a
sibling of `pass`, which inverts the meaning -- it is a failure. Choosing red
then forces `runtime` off orange: orange beside red fails both gates
(normal-vision dE 7.1, worse than the 15 floor), so `runtime` takes violet.
That was the only ordering of the five that passed.

`silent` is amber and `pass` is green in every figure, and "became SILENT" in
fig 2 uses the same amber: one meaning, one hue. Slots below 3:1 contrast on
the light surface carry a visible numeric label on every segment -- that is the
required relief, not decoration.

Figures are light-mode only, which is correct for print. A dark variant would
need its own validated steps, not an inverted flip.
