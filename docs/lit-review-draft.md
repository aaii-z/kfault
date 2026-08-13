# Literature review — draft

Style: `docs/writing-style.md`. First person singular. Every subsection ends
on a criticism. Citations use the keys in `writing/kfault.bib`.

---

## 2.1 From code generation to GPU kernels

Large language models write competent code in ordinary programming languages.
Codex established both the capability and the way it is measured, generating
Python from docstrings and scoring it with pass@k \citep{chen2021codex};
AlphaCode reached roughly median human ranking in programming contests
\citep{li2022alphacode}; and open models trained on large code corpora closed
much of the remaining gap \citep{li2023starcoder, roziere2023codellama,
guo2024deepseekcoder}.

CUDA is a harder target. A kernel is written against a specific memory
hierarchy and thread model, so its performance depends on hardware detail that
the source code does not state \citep{kirk2012programming}. What that costs in
expert effort is visible in FlashAttention, an IO-aware attention kernel whose
tiling reduces traffic between high-bandwidth memory and on-chip SRAM
\citep{dao2022flashattention}: \citet{ouyang2025kernelbench} note that it
appeared five years after the Transformer was proposed, and took two further
years to reach a new GPU generation. Domain-specific languages such as Triton
and TileLang exist to hide part of this complexity \citep{tillet2019triton,
wang2025tilelang}, and the same pressure has produced a wider body of work on
language models for high-performance computing, covering domain fine-tuning,
translation between parallel programming models, and performance-aligned
training \citep{ding2023hpcgpt, bitan2025unipar, lv2025hpctranscompile,
nichols2026performance}.

The difference that matters here is not difficulty but verifiability. A Python
function can be checked against the specification written in its docstring. A
CUDA kernel carries no such specification. It is checked against another
implementation, on inputs that somebody chose.

*(≈255 words)*

---

## 2.2 How kernel generation is evaluated

KernelBench \citep{ouyang2025kernelbench} is the standard benchmark for this
task. It supplies 250 PyTorch reference modules across three difficulty levels
and asks the model to return an equivalent module with a custom inline CUDA
kernel. It scores the result with $\mathrm{fast}_p$, the fraction of generated
kernels that are functionally correct and at least $p$ times faster than the
PyTorch baseline. Frontier reasoning models match the baseline in fewer than
20\% of cases, so the benchmark is far from saturated.

The correctness half of that metric is decided by a single procedure. The
authors "evaluate on 5 random inputs per problem", checking "equivalence of
output between reference architecture \texttt{Model} and generated
architecture with custom kernel \texttt{ModelNew}". They justify the choice
carefully: formal equivalence checking is undecidable, so random testing is
"the most common practical approach", and it is "particularly effective for AI
kernels, where control flow is simpler and the focus is primarily on numerical
correctness". Five inputs are reported as "a good tradeoff between the ability
to catch errors and efficiency", supported by an experiment on 100 generated
kernels in which "the 0/5 and 0/100 failures indicate that no partial
correctness was observed".

That experiment is the weak joint. It varies only the values inside inputs of
one fixed shape and memory layout, so a kernel whose fault depends on layout
rather than on values will pass all five draws and all one hundred. The absence
of partial correctness is therefore a property of the sampling procedure, not
evidence that five inputs suffice.

Later benchmarks refine the metric without revisiting that procedure.
CUDABench \citep{zhu2026cudabench} moves to text-to-CUDA generation and reports
a mismatch between high compilation success and low functional correctness.
The closest work to this dissertation is
\citet{lange2025robustkbench}, who state that existing kernel benchmarks
"suffer from exploitable loopholes and insufficient diversity in testing
conditions" and answer with robust-kbench and an LLM-based verifier, and
\citet{li2026cudabeaver}, who show that a model can pass the correctness tests
"with repair by degeneration, simplifying the CUDA code into a safer but slower
program". \citet{jain2025livecodebench} make the parallel argument for general
code benchmarks, where contamination inflates reported ability.

Each of these accepts the same underlying move: a kernel is reduced to one
binary verdict produced by comparing outputs on inputs somebody chose. Lange
and Li both demonstrate that the verdict can be gamed. Neither asks how often
it is simply wrong.

*(≈430 words)*

---

## 2.3 The oracle problem in code benchmarks

The difficulty named at the end of the last section has a name in software
engineering. A test oracle "determines whether a test execution reveals a
fault, often by comparing the observed program output to the expected output",
and the impossibility of building a complete one is "recognised as one of the
fundamental challenges of software testing" \citep{segura2016metamorphic}.
KernelBench's five random inputs are an oracle in exactly this sense. The
question is not whether it is imperfect, which its authors concede, but how
much it misses.

\citet{liu2023evalplus} answer that question for Python. They extend
HumanEval's test cases by a factor of 80 using type-aware mutation and
LLM-generated inputs, and evaluate 26 models on the result. The extended suite
catches "significant amounts of previously undetected wrong code", reducing
pass@k by up to 19.3--28.9\%. More damaging than the drop is its unevenness:
they find that "test insufficiency can lead to mis-ranking", with two models
overtaking ChatGPT on the extended benchmark that could not beat it on the
original. An incomplete oracle does not simply flatter every model equally; it
reorders them.

Metamorphic testing offers a principled way to add such tests when no reference
output is available. Rather than checking a concrete output, it applies "a
transformation to a test input" and observes "how the program output 'morphs'
into a different one as a result" \citep{segura2016metamorphic}. The relation
that matters for a kernel advertised as a drop-in replacement is close to
trivial once stated: presenting the same values through a different memory
layout must not change the result. No kernel benchmark surveyed above applies
it.

\citet{ludwig2026kaizen} show what happens when it is applied nearby. Testing
LLM-translated CUDA-to-OpenMP code with metamorphic fuzzing and differential
testing across 16 scientific applications, they expose divergences in programs
"that compile and pass developer-written tests yet produce incorrect scientific
results", and report that such errors "are often input-dependent and require
differential testing to expose". Their setting is translation rather than
generation, and they run no repair loop, but the failure mode is the same one
this dissertation measures.

The vocabulary for that failure mode already exists outside software testing.
Fleet-scale studies of processors describe corruptions that escape every error
reporting mechanism, where "the only symptom is an erroneous computation"
\citep{hochschild2021cores, dixit2021silent}. I adopt the same word, silent,
for a kernel that satisfies the benchmark and returns the wrong answer.

Taken together this literature makes a claim the kernel benchmarks have not
absorbed: a test suite does not measure correctness, it defines it, and the
definition is incomplete by construction.

*(≈450 words)*

---

## 2.4 Iterative repair

If one attempt fails, ask again with the error attached. \citet{chen2024selfdebug}
formalised this as SELF-DEBUGGING, teaching a model to perform "rubber duck
debugging" by inspecting execution results and explaining its own code, and
report gains of up to 12\% on benchmarks with unit tests, matching baselines
that "generate more than 10$\times$ candidate programs".
\citet{madaan2023selfrefine} generalise it to any task with one model acting as
generator, critic and reviser, improving performance "by $\sim$20\% absolute on
average" across seven tasks. \citet{shinn2023reflexion} add an episodic memory
of verbal reflections and reach "a 91\% pass@1 accuracy on the HumanEval coding
benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80\%".
These three results are why repair loops are now standard.

Three later studies qualify them. \citet{olausson2024selfrepair} charge repair
its true cost, since each turn is another model call, and find that "when the
cost of carrying out repair is taken into account, performance gains are often
modest, vary a lot between subsets of the data, and are sometimes not present
at all". They attribute this to a bottleneck that matters here: repair is
limited by "the model's ability to provide feedback on its own code", and
improves sharply when a stronger model supplies that feedback.
\citet{huang2024cannot} press further and find that without external feedback
models "struggle to self-correct their responses" and "at times, their
performance even degrades". Crucially, they show that reported successes in
earlier work "result from using oracle labels to guide the self-correction
process, and the improvements vanish when oracle labels are not available".
\citet{brown2024monkeys} reach a compatible conclusion from the sampling side:
coverage scales with the number of attempts across four orders of magnitude,
but converts into real performance only "in domains like coding and formal
proofs, where answers can be automatically verified".

The kernel generation systems adopt the pattern wholesale. CUDA-L1
\citep{li2026cudal1} rewards speedup under contrastive reinforcement learning
and reports $\times$3.12 average speedup on KernelBench; Astra
\citep{wei2025astra} runs specialised agents through "iterative code
generation, testing, profiling, and planning"; K-Search \citep{cao2026ksearch}
searches with a co-evolving world model; Kernel-Smith \citep{du2026kernelsmith}
evolves a population using "structured execution feedback on compilation,
correctness, and speedup"; CUDA Agent \citep{dai2026cudaagent} trains with
"automated verification and profiling to provide reliable reward signals"; and
CUDA-LLM \citep{chen2025cudallm} refines against correctness "validated through
extensive and diverse test cases". Reasoning models, which decompose a problem
into intermediate steps before answering \citep{wei2022cot}, are the strongest
performers throughout.

The feedback each system returns is compilation status, a crash, a timing
measurement, or the verdict of Section 2.2. None returns anything about memory
layout. By Olausson's account a repair loop can only be as good as the signal
it receives, and by Huang's account improvement measured against a signal the
model can see is not evidence of correctness. Every system above optimises
against an oracle that Section 2.3 shows to be incomplete, and each reports the
resulting score as its headline result.

*(≈490 words)*

---

## 2.5 Gap statement

Two lines of work meet in this dissertation without having met in the
literature. Every recent CUDA kernel generation system repairs: it returns
compiler errors and failed output comparisons to the model and asks again
\citep{li2026cudal1, wei2025astra, cao2026ksearch, du2026kernelsmith,
dai2026cudaagent}. Software testing has long held that such comparisons are
incomplete oracles, and that adding tests to a benchmark exposes code that was
wrong all along \citep{liu2023evalplus, segura2016metamorphic}.
\citet{lange2025robustkbench} show that KernelBench's oracle is exploitable,
and \citet{li2026cudabeaver} show that a model can satisfy it by degrading the
kernel it was asked to fix. Neither measures what repair does to correctness
across turns. No published work asks whether a repaired kernel becomes correct
or merely becomes harder to catch. I measure that.

*(108 words)*

---

## Notes on the choices above

- **"without having met"** — the gap is a collision between two literatures,
  not an absence in one. Harder to falsify than "nobody has studied X".
- **Lange and Li are named explicitly**, not buried. They are the closest prior
  work; stating their contribution plainly and then drawing the line is
  stronger than hoping the examiner misses them.
- **"correct or merely harder to catch"** — the whole dissertation in six
  words. Reuse this phrasing in the abstract and the conclusion.
- **"I measure that."** — short final sentence after four long ones. Both Liu
  and Lange close paragraphs this way.
- Uses `\citep`/`\citet` (natbib). With plain `apalike` and no natbib, use
  `\cite{...}` and rewrite "\citet{lange2025robustkbench} show" as
  "Lange et al. \cite{lange2025robustkbench} show".
