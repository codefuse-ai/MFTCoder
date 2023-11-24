"""This metric implements code evaluation with execution across multiple languages as used in the paper 
"OctoPack: Instruction Tuning Code Large Language Models" (https://arxiv.org/abs/2308.07124)."""

import itertools
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
import numpy as np

import evaluate

from .execute import check_correctness


_CITATION = """\
@article{muennighoff2023octopack,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal={arXiv preprint arXiv:2308.07124},
      year={2023}
}
"""

_DESCRIPTION = """\
This metric implements code evaluation with execution across multiple languages as used in the paper 
"OctoPack: Instruction Tuning Code Large Language Models" (https://arxiv.org/abs/2308.07124).
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of candidates to evaluate. Each candidates should be a list
        of strings with several code candidates to solve the problem.
    references: a list with a test for each prediction. Each test should evaluate the
        correctness of a code candidate.
    k: number of code candidates to consider in the evaluation (Default: [1, 10, 100])
    num_workers: number of workers used to evaluate the canidate programs (Default: 4).
    timeout:
Returns:
    pass_at_k: dict with pass rates for each k
    results: dict with granular results of each unittest
Examples:
    >>> code_eval = evaluate.load("code_eval")
    >>> test_cases = ["assert add(2,3)==5"]
    >>> candidates = [["def add(a,b): return a*b", "def add(a, b): return a+b"]]
    >>> pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1, 2])
    >>> print(pass_at_k)
    {'pass@1': 0.5, 'pass@2': 1.0}
"""


_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""

_LICENSE = """The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""


# https://github.com/THUDM/CodeGeeX/blob/ebeb850f227a90c79de39f7e26b1302f374f3240/codegeex/benchmark/rust/Cargo.toml
BASE_CARGO = '''[package]
name = "rust"
version = "0.1.0"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
rand = "0.4"
regex = "1"
md5 = "0.7.0"
'''

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CodeEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Value("string"),
                }
            ),
            homepage="https://github.com/openai/human-eval",
            codebase_urls=["https://github.com/openai/human-eval"],
            reference_urls=["https://github.com/openai/human-eval"],
            license=_LICENSE,
        )

    def _compute(self, predictions, references, k=[1, 10, 100], num_workers=4, timeout=3.0, language="python", cargo_string=BASE_CARGO):
        """Returns the scores"""

        if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
            raise ValueError(_WARNING)

        if os.name == "nt":
            raise NotImplementedError("This metric is currently not supported on Windows.")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
                for candidate in candidates:
                    test_program = candidate + "\n" + test_case
                    args = (test_program, timeout, task_id, completion_id[task_id], language, cargo_string)
                    future = executor.submit(check_correctness, *args)
                    futures.append(future)
                    completion_id[task_id] += 1
                    n_samples += 1

            for future in as_completed(futures):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

        return pass_at_k, results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
