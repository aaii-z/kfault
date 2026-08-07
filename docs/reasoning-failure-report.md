# Where the Thinking Model Goes Wrong

A simple report on the "think" arm (DeepSeek V3.2 with reasoning ON). It reads
the reasoning text the model wrote before its code, and shows the exact spot
where the thinking broke. Every bucket has a real example from our runs.

Format for each bucket: **Report** (very simple), **Details** (what really
happened in the code), **Reasoning** (the wrong thought that caused it).

---

## Bucket A: It thinks hard about the hard part and forgets the easy part

**Report (like for a 5 year old):**
Imagine a kid who builds a big LEGO castle with great care. The castle is
perfect. But he forgets to open the front door, so nobody can go inside. The
hard part is great. One tiny easy part is missing, and the whole thing fails.

**Details (real example, problems 28, 29, 72):**
The model writes a big correct CUDA kernel. Then, in the normal Python part, it
copies the way PyTorch sets up a Linear layer:
```python
nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
bound = 1 / sqrt(fan_in)
```
It uses `sqrt(...)` in Python but never writes `import math`. So Python stops
with `name 'sqrt' is not defined`. The kernel never even runs.

**Reasoning (the wrong part):**
The reasoning text talks a lot about the kernel math. It says nothing about
this setup code. The model wrote that setup code from memory, like a habit, and
did not check it. The problem is not a wrong thought. The problem is a missing
thought. All the thinking went to the kernel, and none went to the plain
Python around it.

---

## Bucket B: It has one clever idea that breaks the rules

**Report (like for a 5 year old):**
A kid sees a toy box has a lid that is never used. He says "this lid is
useless" and throws it away. He is kind of right. But the rule was that every
box must have a lid to fit on the shelf. Now his box does not fit. His smart
idea broke a rule he did not know about.

**Details (real example, problem 26):**
The test always builds the model like this, with 7 values:
```python
ModelNew(in_channels, out_channels, kernel_size, stride, padding,
         output_padding, bias_shape)
```
The model noticed that `self.bias` is never used in `forward`. So it removed
`bias_shape` from the model setup, leaving only 6 slots. When the test sends 7
values into 6 slots, Python fails:
`ModelNew.__init__() takes 7 positional arguments but 8 were given`.

**Reasoning (the wrong part):**
The reasoning text really says this: "the forward does not use self.bias". That
is a true and smart observation. But it acted on it and dropped the input. It
did not know the test has a fixed shape and always passes the same 7 values.
So the wrong step is a real thought that was locally correct but broke a hidden
rule of the benchmark.

---

## Bucket C: It asks for a tool that is not on the table

**Report (like for a 5 year old):**
A kid wants to cut paper and says "I will use the big electric machine in the
factory." But there is no factory here, only the small table. He planned to use
a tool that is not there, so he cannot start.

**Details (real example, problem 4):**
The model wrote CUDA code that calls NVIDIA's cuDNN library:
```cpp
#include <cudnn.h>
```
But our build uses `load_inline`, which does not link cuDNN. So the build
fails: `Error building extension 'fused_conv_double_mish'`.

**Reasoning (the wrong part):**
The model assumed a heavy outside library is ready to use. It did not check what
the simple inline build actually gives. The wrong step is a false assumption
about the tools available, made before writing any code.

---

## Bucket D: The code runs and looks correct, but the answer is secretly wrong

**Report (like for a 5 year old):**
A kid learns to tie his shoe, but only when the shoe sits flat and straight the
normal way. When you turn the shoe a little, he ties it wrong, but it still
looks tied. From far away it looks fine. Up close it is wrong.

**Details (real example, problem 14, "silent" failure):**
The kernel reads numbers from memory with a fixed straight formula:
```cpp
x_shared[k] = x[i * input_size + k];
```
This only works if the data sits in one straight line in memory (contiguous,
row by row). On the normal test input, the data is straight, so the answer is
correct and the kernel "passes". But when we give it the same numbers in a
different memory shape (non contiguous), the formula reads the wrong spots and
the answer is wrong. The bug was hidden on the normal test and only showed
under our stress battery.

**Reasoning (the wrong part):**
The reasoning text shows the model saw this risk and then waved it away. It
wrote: "This might be okay if the weight matrix is stored in row-major order."
It made a hope into a fact. It assumed the memory is always straight instead of
handling any layout. So the wrong step is an unchecked assumption about memory,
which is exactly the kind of bug a standard test does not catch.

---

## The one big pattern

Reasoning ON does not make the model careless about the math. It makes the math
mostly good. The failures move to the parts the reasoning did not cover:

- the plain setup code (Bucket A)
- the fixed rules of the benchmark interface (Bucket B)
- the tools the build really has (Bucket C)
- quiet assumptions about memory and inputs (Bucket D)

Buckets A, B, C stop the code from running, so they are easy to see. Bucket D
is the dangerous one. The code runs, the normal test passes, and the wrong
answer stays hidden. This is the "silent failure" at the heart of the study.
