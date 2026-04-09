#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "litellm",
#     "tqdm",
#     "datasets==4.0.*",
# ]
# ///
"""
This script evaluates models on the CanItEdit dataset using vLLM for hosting
and LiteLLM as the client for inference.
"""

import datasets
from pathlib import Path
from tqdm import tqdm
from typing import List, Literal, Optional, TypedDict, Callable, TypeVar
import gzip
import json
from litellm import atext_completion, acompletion
import itertools
import asyncio


def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


T = TypeVar("T")

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    # the prefix for the assistant's response. this is only used for the
    # last message in a conversation, and is ignored otherwise.
    # NOTE: leaving commented for python 3.10 compatibility
    #  prefix_after: NotRequired[str]


# this is one edit request
class EditCommand(TypedDict):
    instruction: Optional[str]
    content: str


# this is model's output
class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


# (old, instr, new) -> prompt
PromptFormatFunction = Callable[[str, str, str], str]

# (old, instr) -> [messages]
MessagesFormatFunction = Callable[[str, str], List[Message]]

# (old, new) -> response
PostProcessFunction = Callable[[str, str], str]


def chat_edit_prompt_zeroshot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
            """.strip(),
        },
        {
            "role": "user",
            "content": f"""
## Code Before
```py
{old}
```

## Instruction
{instr}

## Code After""".strip(),
        },
    ]



# Copied from prl_ml
def extract_code_from_markdown(markdown):
    """
    Extracts the first markdown block of code from markdown.

    Strips away the language tag on the first line if present. Supports markdown
    that has several code blocks (just returns the first).
    """
    # Find the first code block
    code_block_start = markdown.find("```")
    if code_block_start == -1:
        return None
    
    # Skip past the opening ```
    code_start = code_block_start + 3
    
    # Find the end of this code block
    code_block_end = markdown.find("```", code_start)
    if code_block_end == -1:
        return None
        
    # Extract the code between the markers
    code = markdown[code_start:code_block_end].strip()

    if "# Example usage:" in code:
        code = code.split("# Example usage:")[0]
    
    # Remove language tag if present on first line
    first_newline = code.find('\n')
    if first_newline > 0:
        # Consider the case where the block begins with "```python\n...". In this
        # case, code would already be "python\n..." and first_newline would be 7.
        # Thus first_newline + 1 is the index of "...".
        code = code[first_newline + 1:]
            
    return code.strip()


# Note that the code below is not fenced.
ONE_SHOT_EXAMPLE = {
    "before": """def add(a, b):
    return a + b""",
    "instruction": """Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.""",
    "after": '''def add(x, y):
    """Adds two numbers."""
    return x + y

def sub(x, y):
    """Subtracts two numbers."""
    return x - y'''
}

class EditModel:
    def __init__(self):
        pass

    async def generate(self, prompt: EditCommand, **kwargs) -> EditResponse:
        raise NotImplementedError

    def get_prompt_format(self):
        raise NotImplementedError



class DirectEditModel(EditModel):
    """
    The direct kind of edit model, this class is supposed to be used either with EditCoder or
    with non-chat models, like foundation models.
    """

    def __init__(self, model_name: str, one_shot: bool):
        super().__init__()
        self.model_name = model_name
        self.one_shot = one_shot

    def format_prompt(
        self,
        old,
        instr,
        codeblock_before: Optional[str] = None,
    ):
        """
        The codeblock_before and codeblock_after arguments are used to specify
        if there should be a codeblock surrounding the code before and after
        the instruction. If None, then no codeblock is used. The string is the
        extension of the codeblock, e.g. "py" or "md".
        """
        if codeblock_before is not None:
            old = f"```{codeblock_before}\n{old}\n```"
        before = f"""## Code Before:\n{old}\n"""
        instr = f"""## Instruction:\n{instr}\n"""
        after = f"""## Code After:\n"""
        prompt = before + instr + after
        if self.one_shot:
            prompt = f"## Code Before\n{ONE_SHOT_EXAMPLE['before']}\n## Instruction:\n{ONE_SHOT_EXAMPLE['instruction']}\n## Code After\n{ONE_SHOT_EXAMPLE['after']}\n{prompt}"
        return prompt

    def extract_from_response(self, response_text: str) -> str:
        code = extract_code_from_markdown(response_text)
        if code is not None:
            return code
        return response_text

    def get_stop_tokens(self):
        # NOTE(arjun): These are the original stop tokens from the CanItEdit
        # code, which you can verify here:
        #
        # https://github.com/nuprl/CanItEdit/blob/1a87cb488e7ff801cce80550e20822228e1c88fe/benchmark/generate_completions.py#L507
        #
        # However, notice that in the paper and the few-shot prompt, there
        # may not be a ":" after the stop tokens. However, if you see
        # the training code, this is how EditCoder was trained:
        #
        # https://github.com/nuprl/CanItEdit/blob/1a87cb488e7ff801cce80550e20822228e1c88fe/editpackft/format.py#L20
        return [
            "## Code After:",
            "## Instruction:",
            "## Code Before:",
            "## Test Case:",
            "## Explanation:"
        ]

    async def generate(self, prompt: EditCommand, **kwargs) -> EditResponse:
        assert prompt["instruction"] is not None, "Not implemented yet"
        str_prompt = self.format_prompt(prompt["content"], prompt["instruction"])

        # Generate using text_completion directly
        response = await atext_completion(
            model=self.model_name,
            prompt=str_prompt,
            **kwargs,
            stop=self.get_stop_tokens(),
        )
        return self.extract_from_response(response.choices[0].text)


class AgentPackEditModel(DirectEditModel):

    def __init__(self, model_name: str, one_shot: bool):
        super().__init__(model_name, one_shot)
    
    def format_prompt(
        self,
        old,
        instr,
        codeblock_before: Optional[str] = None,
        codeblock_after: Optional[str] = None,
    ):
        prompt = f"# Code Before:\n\n```\n{old.rstrip()}\n```\n\n# Instruction:\n\n{instr}\n\n# Code After:\n\n```\n"
        if self.one_shot:
            prompt = f"# Code Before:\n\n```\n{ONE_SHOT_EXAMPLE['before']}\n```\n\n# Instruction:\n\n{ONE_SHOT_EXAMPLE['instruction']}\n\n# Code After:\n\n```\n{ONE_SHOT_EXAMPLE['after']}\n```\n\n{prompt}"
        return prompt
    
    def extract_from_response(self, response_text: str) -> str:
        return response_text
    
    def get_stop_tokens(self):
        return [
            "```",
        ]


class ChatAdaptorEditModel(EditModel):
    """
    This is an adaptor class to use ChatModels as EditModels.
    NOTE: This model class is only intended for inference, not training.
    """

    def __init__(
        self,
        model_name,
    ):
        super().__init__()
        self.model_name = model_name

    async def generate(self, prompt: EditCommand, **kwargs) -> EditResponse:
        assert prompt["instruction"] is not None, (
            "Every command must have an instruction in ChatAdaptorEditModel"
        )
        response = await acompletion(
            model=self.model_name,
            messages=chat_edit_prompt_zeroshot(prompt["content"], prompt["instruction"]),
            **kwargs,
        )
        gen = response.choices[0].message.content
        code = extract_code_from_markdown(gen)
        if code is not None:
            return code
        return gen


async def process_example_and_instruction(
    ex: dict,
    instr_kind: str,
    model: EditModel,
    model_kwargs: dict,
    args,
    output_dir: Path,
    request_sema: asyncio.Semaphore,
    pbar: tqdm,
) -> None:
    """
    Process a single example and instruction kind by generating completions and saving results.
    Uses a request-level semaphore so that a fixed number of requests are always
    in-flight, keeping the GPU saturated even when some requests finish early.
    """
    # TODO(arjun): Not real resume. :)
    path = output_dir / f"{ex['full_name']}_{instr_kind}.json.gz"
    if path.exists():
        pbar.update(1)
        return  # this pretty much resumes from where it left off

    example = EditCommand(instruction=ex[instr_kind], content=ex["before"])

    async def gen(example):
        async with request_sema:
            try:
                async with asyncio.timeout(600):
                    return await model.generate(example, **model_kwargs)
            except asyncio.TimeoutError:
                return "<timeout encountered>"

    completion_tasks = [gen(example) for _ in range(args.completion_limit)]
    completions = await asyncio.gather(*completion_tasks)

    # copy over the example
    result = {}
    for k in ex:
        result[k] = ex[k]

    result["instr_kind"] = instr_kind
    # this is for compatibility with the MultiPL-E evaluator
    result["prompt"] = ""
    result["completions"] = completions
    result["language"] = "py"
    result["temperature"] = args.temperature
    result["top_p"] = args.top_p
    result["max_tokens"] = args.max_tokens
    result["stop_tokens"] = getattr(model, "stop_tokens", [])
    result["script_args"] = args.__dict__.copy()

    gunzip_json_write(path, result)
    pbar.update(1)


async def main(args):
    dataset = datasets.load_dataset(args.dataset, args.subset, split=args.split)

    # Direct model instantiation based on model_type
    if args.model_type == "editcoder":
        model = DirectEditModel(args.model, False)
    elif args.model_type == "editcoder-1shot":
        model = DirectEditModel(args.model, True)
    elif args.model_type == "agentpack":
        model = AgentPackEditModel(args.model, False)
    elif args.model_type == "agentpack-1shot":
        model = AgentPackEditModel(args.model, True)
    elif args.model_type == "chat":
        model = ChatAdaptorEditModel(args.model)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    instr_kinds = ["instruction_descriptive", "instruction_lazy"]
    items = list(itertools.product(dataset, instr_kinds))

    # Request-level concurrency: keep exactly batch_size requests in-flight
    # at all times. As individual requests complete, new ones immediately
    # fill the slot — no waiting for an entire item's completions to finish.
    request_sema = asyncio.Semaphore(args.batch_size)

    pbar = tqdm(total=len(items))

    async with asyncio.TaskGroup() as tg:
        for ex, instr_kind in items:
            tg.create_task(
                process_example_and_instruction(
                    ex,
                    instr_kind,
                    model,
                    model_kwargs,
                    args,
                    Path(args.output_dir),
                    request_sema=request_sema,
                    pbar=pbar,
                )
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="nuprl/CanItEdit", help="dataset to use"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="split of the dataset to use"
    )
    parser.add_argument(
        "--subset", type=str, default=None, help="subset of the split to use"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["editcoder", "editcoder-1shot", "agentpack","agentpack-1shot","chat", "chat_oneshot"],
        help="type of model to use for completions",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="path to model or hub name"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="output directory for completions"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="batch size for completions"
    )
    parser.add_argument(
        "--completion-limit",
        type=int,
        default=20,
        help="number of completions per prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p sampling")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="max new tokens to generate per completion. 2048 works for CanItEdit",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
