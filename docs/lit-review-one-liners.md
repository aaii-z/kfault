# One-sentence summaries — all 35 papers

`[nnn]` = folder in `files/`. Summaries taken from the papers' own abstracts.

## Kernel generation with LLMs

- **[185] CUDA-LLM** (Chen et al., 2025) — Proposes Feature Search and Reinforcement, a loop that jointly optimises compilation, functional correctness and measured kernel latency, reporting speedups up to 179× over human-written code.
- **[202] CUDA-L1** (Li et al., ICLR 2026) — A contrastive reinforcement learning framework that reaches ×3.12 average speedup across all 250 KernelBench kernels using speedup-based rewards alone, and transfers across five GPU architectures.
- **[257] Kernel-Smith** (Du et al., 2026) — Combines an evolutionary agent maintaining a population of executable candidates with an evolution-oriented post-training recipe, reaching state-of-the-art average speedup on KernelBench's Triton backend.
- **[624] K-Search** (Cao et al., 2026) — Replaces static search heuristics with a co-evolving world model that separates high-level planning from low-level code generation, achieving 2.10× average improvement on FlashInfer kernels.
- **[625] Astra** (Wei et al., 2025) — The first multi-agent LLM system for kernel optimisation, starting from real SGLang CUDA implementations rather than PyTorch modules and reaching 1.32× average speedup zero-shot.
- **[705] CUDA Agent** (Dai et al., 2026) — A large-scale agentic RL system with data synthesis, automated verification and profiling rewards that beats `torch.compile` on 100%/100%/92% of KernelBench Levels 1–3.

## Benchmarks, evaluation and verification

- **[435] KernelBench** (Ouyang et al., ICML 2025) — Introduces a 250-task benchmark and the `fast_p` metric for kernels that are both correct and faster than baseline, finding frontier reasoning models match PyTorch in under 20% of cases.
- **[232] robust-kbench** (Lange et al., 2025) — Argues existing kernel benchmarks have exploitable loopholes and insufficient testing diversity, and pairs a stricter benchmark with an agentic discovery-verification-optimisation pipeline.
- **[729] CUDABeaver** (Li et al., 2026) — A benchmark of 213 real failing CUDA workspaces that exposes "repair by degeneration", where a model passes correctness tests by simplifying code into a safer but slower program.
- **[737] CUDABench** (Zhu et al., 2026) — A text-to-CUDA benchmark across AI, scientific computing and data analytics that finds a notable mismatch between high compilation success and low functional correctness.
- **[561] LiveCodeBench** (Jain et al., 2024) — A contamination-free code benchmark built from continuously released competition problems, exposing overfitting that static benchmarks hide.
- **[627] Codex / HumanEval** (Chen et al., 2021) — Introduces HumanEval and the pass@k metric, establishing the standard way LLM code generation has been measured ever since.

## Weak oracles and test inadequacy

- **[731] EvalPlus** (Liu et al., NeurIPS 2023) — Adding 80× more test cases to HumanEval cut pass@k by 19–28% across 26 models and reordered them, showing benchmark tests are too weak to establish correctness.
- **[733] Kaizen** (Ludwig et al., 2026) — Metamorphic fuzzing and differential testing of LLM-translated CUDA→OpenMP code across 16 applications found semantic errors that compile and pass developer-written tests yet return wrong scientific results.
- **[738] Metamorphic testing survey** (Segura et al., IEEE TSE 2016) — Surveys metamorphic testing, which sidesteps the oracle problem by checking how output changes under an input transformation instead of predicting the correct output.
- **[739] Silent Data Corruptions at Scale** (Dixit et al., 2021) — Facebook found hundreds of CPUs silently producing wrong results across its fleet over 18 months, none caught by hardware error reporting.
- **[741] Cores That Don't Count** (Hochschild et al., HotOS 2021) — Google reports "mercurial" cores whose only symptom is an erroneous computation, breaking the fail-stop assumption most system software relies on.

## Repair and self-correction

- **[735] Self-Debug** (Chen et al., 2023) — Teaching a model to inspect execution results and explain its own code raises accuracy by up to 12% and matches baselines sampling 10× more candidates.
- **[736] Self-Refine** (Madaan et al., NeurIPS 2023) — One model acting as generator, critic and reviser improves performance by roughly 20% absolute across 7 tasks with no training.
- **[740] Reflexion** (Shinn et al., 2023) — Agents storing verbal reflections on failures in episodic memory reach 91% pass@1 on HumanEval versus 80% for GPT-4.
- **[732] Is Self-Repair a Silver Bullet?** (Olausson et al., ICLR 2024) — Once extra compute cost is counted, self-repair gains are modest or absent, because repair is bottlenecked by the model's ability to judge its own code.
- **[734] LLMs Cannot Self-Correct Reasoning Yet** (Huang et al., ICLR 2024) — Without external feedback LLMs fail to self-correct and often get worse, and earlier positive results came from oracle labels that vanish when removed.
- **[546] Large Language Monkeys** (Brown et al., 2024) — Coverage scales log-linearly with sample count over four orders of magnitude, but only converts into real performance where an automatic verifier exists; majority voting and reward models plateau.
- **[742] Chain-of-Thought** (Wei et al., NeurIPS 2022) — Prompting for intermediate reasoning steps sharply improves complex reasoning, an ability that emerges only in sufficiently large models.

## Kernel programming models and systems

- **[583] Triton** (Tillet et al., MAPL 2019) — An intermediate language and compiler built on tiled computation that lets developers write neural network kernels competitive with hand-tuned CUDA.
- **[607] TileLang** (Wang et al., 2025) — A tiled programming model that decouples the scheduling space (thread binding, layout, pipelining) from dataflow, so users describe dataflow and the compiler handles the rest.
- **[462] FlashAttention** (Dao et al., NeurIPS 2022) — An IO-aware exact attention algorithm using tiling to cut reads and writes between HBM and SRAM, giving 3× speedup on GPT-2 and enabling far longer context.

## LLMs for HPC and code translation

- **[247] HPCTransCompile** (Lv et al., 2025) — Builds an AI-compiler-generated dataset for translating CUDA to other platforms, addressing the performance-portability problem created by CUDA's dominance.
- **[610] UniPar** (Bitan et al., 2025) — A systematic evaluation framework for LLM-based translation between serial code, CUDA and OpenMP across four usage modes.
- **[611] HPC-GPT** (Ding et al., SC-W 2023) — A LLaMA model fine-tuned on generated HPC question-answer pairs, evaluated on managing AI models/datasets and on data race detection.
- **[629] Performance-Aligned LLMs** (Nichols et al., TPDS 2026) — Uses reinforcement learning to align code LLM outputs with measured performance, raising expected speedup from 0.9 to 1.6 for serial code and 1.9 to 4.5 for OpenMP.

## Foundation code LLMs

- **[615] Code Llama** (Rozière et al., 2023) — A family of open code models built on Llama 2 with long-context handling, infilling and instruction following.
- **[613] DeepSeek-Coder** (Guo et al., 2024) — Open code models trained from scratch on 2T tokens with repository-level context and fill-in-the-middle, closing much of the gap to closed models.
- **[575] StarCoder** (Li et al., TMLR 2023) — An openly licensed 15B code model trained on permissively licensed GitHub data with an 8K context and attribution tooling.
- **[617] AlphaCode** (Li et al., Science 2022) — Reaches roughly median human ranking in competitive programming contests through massive sampling followed by filtering and clustering.
