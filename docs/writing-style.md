# Writing style

Extracted from `writing-style/`: Jiawei Liu (EvalPlus, EvalPerf) and Robert
Tjarko Lange (robust-kbench, Evolution Transformer, LLMs as Evolution
Strategies). Both write the same way.

**One adaptation.** Both papers are multi-author and say "we". This
dissertation has one author, so write **"I"** throughout: *I introduce*,
*I find*, *I chose*. Never "we", never "the author".

---

## The eight rules

**1. Name the thing, then say what is wrong with the alternative.**
Liu: *"We introduce Differential Performance Evaluation (DPE), a framework
designed to reliably evaluate Large Language Models for efficient code
generation. Traditional coding benchmarks often fail to provide reliable
insights into code efficiency, due to their reliance on simplistic test
inputs and the absence of effective compound metrics."*

Introduce the contribution, then immediately state the deficiency it answers.
Never open with a history lesson.

**2. Ask the question out loud.**
Lange: *"Here, we set out to answer a simple question: Can we leverage recent
agentic advances to improve the robust discovery, self-verification, and
optimization of low-level CUDA operations?"*

One explicit question in the introduction. It orients the whole report.

**3. Show a concrete failure before explaining it.**
EvalPlus Figure 1 is real wrong code that ChatGPT produced, printed beside the
input that breaks it. The reader sees the phenomenon before reading any theory
about it.

**4. Contributions as a numbered list with bold run-in headings.**
Lange: *"1. **A Robust Benchmark Harness:** We highlight several pitfalls of
LLM-driven CUDA kernel optimization, including the exploitation of bad task
design and narrow verification leading to artificial speedup estimates."*

Four or five items. Each starts with a noun phrase naming the artefact, not a
verb.

**5. Figure 1 is a system overview whose caption narrates it.**
Lange captions with *Left / Middle / Right*, walking through the pipeline end
to end, so the figure is readable without the body text.

**6. State negative findings flatly.**
Lange: *"we also find detrimental shortcomings in the current state of
benchmarking LLM-written CUDA kernels"*. No softening, no apology. Let the
data carry the claim.

**7. Concede the ideal, then explain why it is unreachable.**
Liu: *"Ideally, we would like to formally verify the correctness of
LLM-provided solutions for any input, but verifying domain-specific problems
through methods such as translation validation is already challenging enough,
let alone building a general verifier with absolute certainty."*

This is how you justify empirical testing instead of proof.

**8. Active voice, first person, short declarative sentences.**
"I find", not "it can be observed that". One idea per sentence.

---

## What to avoid

- **"In recent years, ..."** — neither author ever opens this way.
- **Hedging stacks**: "may potentially suggest that". Write "shows".
- **Passive constructions that hide the actor**: "it was decided that".
  Write "I chose".
- **"We"** — wrong for a single-author dissertation.
- **Sections that end with a summary.** End with a criticism or a consequence.
- **Author-by-author literature recitals.** Group paragraphs by claim, not by
  paper. "Three systems report speedups without checking generalisation
  [a, b, c]" beats three separate paragraphs.
- **Long noun chains**: "kernel correctness evaluation methodology framework".
  Cut to the shortest phrase that still means it.
