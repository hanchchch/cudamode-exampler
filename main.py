import logging

from src.example_models import models
from src.llm import LLM, Input

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_functions(reference_model: str):
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
    with open(f"outputs/learn_from_{reference_model}.md", "w") as f:
        for response in llm.generate(i):
            md += response
            f.write(response)

    snippets = [s for s in md.split("###") if s != ""]
    logger.info(f"Generated {len(snippets)} snippets")
    for snippet in snippets:
        try:
            name = snippet.split("\n")[0].strip().replace(" ", "_").lower()
            code = snippet.split("```python")[1].split("```")[0].strip()
            with open(f"outputs/{reference_model}_{name}.py", "w") as f:
                f.write(code)
        except IndexError:
            logger.warning(f"Failed to parse snippet, model output: {snippet}")
            continue

    return llm.usage


u = write_functions("albert")
print(u)
