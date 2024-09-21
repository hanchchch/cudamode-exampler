import logging
import os
import sys

from src.example_models import models
from src.llm import LLM, Input

output_dir = "outputs"

prompt_write_functions = """These are example pytorch code.
Try writing new pytorch code snippets.
It should be a function that takes a dictionary of torch tensors and returns a torch tensor.

Make it self-contained and independent of other functions.
Every functions or classes should be defined in the code.

The answer format should be like below:
### <summary of the function>
```python
<function code>
```
"""


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def write_functions(reference_model: str):
    md_path = f"{output_dir}/learn_from_{reference_model}.md"
    if os.path.exists(md_path):
        logger.info(f"{reference_model} has already been processed, skipping")
        return

    i = Input()
    llm = LLM()

    modelfiles = models[reference_model]

    for filename, content in modelfiles.items():
        prompt = f"file: {filename}\n{content}"
        i.add(prompt)
    i.add(prompt_write_functions)

    logger.info(
        f"Generating functions based on {reference_model} ({len(modelfiles)} files)"
    )
    md = ""
    with open(md_path, "w") as f:
        for response in llm.generate(i):
            md += response
            f.write(response)

    logger.info(f"Generated markdown file: {md_path} (token usage: {llm.usage})")

    snippets = [s for s in md.split("###") if s != ""]
    logger.info(f"Generated {len(snippets)} snippets")
    for snippet in snippets:
        try:
            name = snippet.split("\n")[0].strip().replace(" ", "_").lower()
            code = snippet.split("```python")[1].split("```")[0].strip()
            with open(f"{output_dir}/{reference_model}_{name}.py", "w") as f:
                f.write(code)
        except IndexError:
            logger.warning(f"Failed to parse snippet, model output: {snippet}")
            continue


if __name__ == "__main__":
    for model in models.keys():
        write_functions(model)
