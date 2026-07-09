"""KernelBench: generate a fused CUDA kernel via DeepSeek, benchmark vs PyTorch on Modal GPU."""

import modal

app = modal.App("kfault")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("torch", "ninja", "openai>=1.0.0", "datasets>=2.0.0")
)

WARMUP = 10
RUNS = 5


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("openrouter")],
    timeout=600,
)
def run_pipeline(problem_num: int = 1, model_id: str = "deepseek/deepseek-chat"):
    import os

    import torch
    from datasets import load_dataset
    from openai import OpenAI

    print(f"Loading KernelBench Level 1 problem {problem_num}...")
    ds = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
    problem = next(p for p in ds if p["name"].startswith(f"{problem_num}_"))
    code, name = problem["code"], problem["name"]
    print(f"Problem: {name}\n")

    # build model and inputs from problem code
    problem_ns: dict = {}
    exec(code, problem_ns)
    Model = problem_ns["Model"]
    get_init_inputs = problem_ns.get("get_init_inputs", lambda: [])
    get_inputs = problem_ns.get("get_inputs", lambda: [])

    init_inputs = get_init_inputs()
    model = Model(*init_inputs).cuda().eval()
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]

    # time pytorch baseline
    with torch.no_grad():
        for _ in range(WARMUP):
            pytorch_out = model(*inputs)
        torch.cuda.synchronize()  # ensure warmup is done before timing starts

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(RUNS):
            pytorch_out = model(*inputs)
        t1.record()
        torch.cuda.synchronize()
        pytorch_ms = t0.elapsed_time(t1) / RUNS

    print(f"PyTorch: {pytorch_ms:.3f} ms/run\n")

    # ask model for a Python file with ModelNew using load_inline
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    prompt = (
        "You are a CUDA expert. Given the following PyTorch model, "
        "write a Python file that replaces it with a custom fused CUDA kernel.\n\n"
        "Requirements:\n"
        "- Use torch.utils.cpp_extension.load_inline to compile and load the kernel\n"
        "- Put PYBIND11_MODULE in the cuda_sources string to expose functions, and do NOT pass the functions= argument to load_inline\n"
        "- Store the returned module in a variable and call functions on it directly, like: ext = load_inline(name=..., cpp_sources='', cuda_sources=src, verbose=True); ext.my_func(x)\n"
        "- Do NOT use is_python_module=False or torch.ops\n"
        "- Define class ModelNew(nn.Module) with the same __init__ and forward signature as Model\n"
        "- Return ONLY the Python code, no explanation, no markdown\n\n"
        f"{code}"
    )
    print(f"Querying {model_id} via OpenRouter...")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw_response = response.choices[0].message.content
    generated_code = raw_response

    # strip markdown code fences if present
    if "```" in generated_code:
        blocks = generated_code.split("```")
        fenced = blocks[1]
        generated_code = fenced[fenced.index("\n") + 1:] if "\n" in fenced else fenced

    print(f"Generated {len(generated_code)} chars\n")

    # try to run generated kernel and benchmark
    kernel_ms = None
    correct = None
    error = None

    try:
        kernel_ns: dict = {}
        exec(generated_code, kernel_ns)
        ModelNew = kernel_ns["ModelNew"]
        model_new = ModelNew(*init_inputs).cuda().eval()

        with torch.no_grad():
            for _ in range(WARMUP):
                kernel_out = model_new(*inputs)
            torch.cuda.synchronize()  # ensure warmup is done before timing starts

            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(RUNS):
                kernel_out = model_new(*inputs)
            t1.record()
            torch.cuda.synchronize()
            kernel_ms = t0.elapsed_time(t1) / RUNS

        correct = torch.allclose(
            pytorch_out.float(), kernel_out.float(), atol=1e-3, rtol=1e-3
        )
        print(f"Kernel:  {kernel_ms:.3f} ms/run")
        print(f"Speedup: {pytorch_ms / kernel_ms:.2f}x")
        print(f"Correct: {correct}")
    except Exception as e:
        error = str(e)
        print(f"Kernel failed: {error[:300]}")

    speedup = pytorch_ms / kernel_ms if kernel_ms else None
    return {
        "problem_name": name,
        "model_id": model_id,
        "raw_response": raw_response,
        "pytorch_ms": pytorch_ms,
        "kernel_ms": kernel_ms,
        "speedup": speedup,
        "correct": correct,
        "error": error,
    }


@app.local_entrypoint()
def main():
    import datetime
    import os

    os.makedirs("results", exist_ok=True)

    import itertools

    # reasoning (r1) vs non-reasoning (v3): same base model, cleanest contrast
    models = {"v3": "deepseek/deepseek-chat", "r1": "deepseek/deepseek-r1"}

    for num, (tag, model_id) in itertools.product([1], models.items()):
        result = run_pipeline.remote(num, model_id)
        name = result["problem_name"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{name}_{tag}_{timestamp}.txt"

        print(f"\n{name} [{model_id}]")
        print(f"  PyTorch : {result['pytorch_ms']:.3f} ms")
        if result["kernel_ms"]:
            print(f"  Kernel  : {result['kernel_ms']:.3f} ms")
            print(f"  Speedup : {result['speedup']:.2f}x")
            print(f"  Correct : {result['correct']}")
        else:
            print(f"  Kernel failed: {result['error'][:1000] if result['error'] else 'unknown'}")

        with open(filename, "w") as f:
            f.write(f"Problem: {name}\n")
            f.write(f"PyTorch : {result['pytorch_ms']:.3f} ms/run\n")
            if result["kernel_ms"]:
                f.write(f"Kernel  : {result['kernel_ms']:.3f} ms/run\n")
                f.write(f"Speedup : {result['speedup']:.2f}x\n")
                f.write(f"Correct : {result['correct']}\n")
            if result["error"]:
                f.write(f"\nError:\n{result['error']}\n")
            f.write("\nFull Model Response:\n")
            f.write(result["raw_response"])

        print(f"  Saved to {filename}")
