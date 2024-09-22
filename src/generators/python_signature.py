import logging

from src.llm import LLM, Input

prompt_write_signature = """Read a function, and write a function signature for that.
It should specify the function's name, parameter's names, dimensions for the tensors, and data types.

It should be in the following format:
```python
function_signature = [
    '<function_name>',
    ('<parameter_name>', <dimension_tuple>, <dtype>),
]
```

For example, consider the following function:
```python
def linear_transformation_activation(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    return torch.relu(output).to(torch.float32)
```
    
The function signature for the above function would be:
```python
function_signature = [
    'linear_transformation_activation',
    ('input_tensor', (4, 4), torch.float32),
    ('weight', (4, 4), torch.float32)
]
```

Now, write a signature for below function. Do not repeat the function, just the signature.
Again, it should specify the function's name, parameter's names, dimensions for the tensors, and data types.
```python
{function}
```
"""


logger = logging.getLogger()


def write_signature(filepath: str):
    logger.info(f"Generating function signature for {filepath}")

    i = Input()
    llm = LLM()

    with open(filepath, "r") as f:
        content = f.read()
        i.add(prompt_write_signature.format(function=content))

    generated = ""
    for response in llm.generate(i):
        generated += response

    try:
        signature = generated.split("```python")[1].split("```")[0].strip()
    except IndexError:
        logger.warning(f"Failed to generate function signature: {generated}")
        return

    with open(filepath, "a") as f:
        f.write("\n\n\n" + signature)

    logger.info(
        f"Generated function signature for {filepath} (token usage: {llm.usage})"
    )
