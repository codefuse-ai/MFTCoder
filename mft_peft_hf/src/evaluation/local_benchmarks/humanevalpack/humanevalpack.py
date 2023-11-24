import json

import datasets


_DESCRIPTION = """
"""

_HOMEPAGE = "https://github.com/bigcode-project/octopack"

def get_url(name):
    url = f"data/{name}/data/humanevalpack.jsonl"
    return url

def split_generator(dl_manager, name):
    downloaded_files = dl_manager.download(get_url(name))
    return [
        datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "filepath": downloaded_files,
            },
        )
    ]

class HumanEvalPackConfig(datasets.BuilderConfig):
    """BuilderConfig """

    def __init__(self, name, description, features, **kwargs):
        super(HumanEvalPackConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.name = name
        self.description = description
        self.features = features


class HumanEvalPack(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        HumanEvalPackConfig(
            name="python",
            description="Python HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),
        HumanEvalPackConfig(
            name="js",
            description="JavaScript HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),        
        HumanEvalPackConfig(
            name="java",
            description="Java HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),
        HumanEvalPackConfig(
            name="go",
            description="Go HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),        
        HumanEvalPackConfig(
            name="cpp",
            description="C++ HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),        
        HumanEvalPackConfig(
            name="rust",
            description="Rust HumanEvalPack",
            features=[
                "task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "import", "test_setup", "test", "example_test", "entry_point", "signature", "docstring", "instruction"
            ]
        ),
    ]
    DEFAULT_CONFIG_NAME = "python"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "task_id": datasets.Value("string"),
                    "prompt": datasets.Value("string"),                  
                    "declaration": datasets.Value("string"),
                    "canonical_solution": datasets.Value("string"),
                    "buggy_solution": datasets.Value("string"),
                    "bug_type": datasets.Value("string"),
                    "failure_symptoms": datasets.Value("string"),
                    "entry_point": datasets.Value("string"),
                    "import": datasets.Value("string"),  
                    "test_setup": datasets.Value("string"),
                    "test": datasets.Value("string"),
                    "example_test": datasets.Value("string"),
                    "signature": datasets.Value("string"),
                    "docstring": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "python":
            return split_generator(dl_manager, self.config.name)

        elif self.config.name == "cpp":
            return split_generator(dl_manager, self.config.name)

        elif self.config.name == "go":
            return split_generator(dl_manager, self.config.name)

        elif self.config.name == "java":
            return split_generator(dl_manager, self.config.name)

        elif self.config.name == "js":
            return split_generator(dl_manager, self.config.name)

        elif self.config.name == "rust":
            return split_generator(dl_manager, self.config.name)
           
    def _generate_examples(self, filepath):
        key = 0
        with open(filepath) as f:
            for line in f:
                row = json.loads(line)
                key += 1
                yield key, {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "declaration": row["declaration"],
                    "canonical_solution": row["canonical_solution"],
                    "buggy_solution": row["buggy_solution"],
                    "bug_type": row["bug_type"],
                    "failure_symptoms": row["failure_symptoms"],
                    "import": row.get("import", ""), # Only for Go                    
                    "test_setup": row.get("test_setup", ""), # Only for Go
                    "test": row["test"],
                    "example_test": row["example_test"],
                    "entry_point": row["entry_point"],
                    "signature": row["signature"],
                    "docstring": row["docstring"],
                    "instruction": row["instruction"],
                }  
                key += 1