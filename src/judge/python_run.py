import runpy
from typing import Callable

import torch


def load_pytorch_function(file_path: str, function_name: str) -> Callable:
    """Load the PyTorch function from the given file."""
    module = runpy.run_path(file_path)
    return module[function_name]


def load_function_signature(
    file_path: str,
) -> tuple[str, list[tuple[str, tuple, torch.dtype]]]:
    """Load the function signature from the given file."""
    module = runpy.run_path(file_path)
    sig = module["function_signature"]
    name = sig["name"]
    args = sig["inputs"]
    # validate the args follow the format
    # [((shape,), dtype)]

    for arg in args:
        if not isinstance(arg, tuple):
            raise ValueError(f"Invalid input signature: {arg}")

        if len(arg) != 2:
            raise ValueError(f"Invalid input signature: {arg}")

        if not isinstance(arg[0], tuple):
            raise ValueError(f"Invalid input shape: {arg}")
        if len(arg[0]) < 1:
            raise ValueError(f"Invalid input shape: {arg}")
        if not all(isinstance(dim, int) for dim in arg[0]):
            raise ValueError(f"Invalid input shape: {arg}")

        if not isinstance(arg[1], torch.dtype):
            raise ValueError(f"Invalid input dtype: {arg}")

    return name, args


def prepare_inputs(signature: list[tuple[tuple, torch.dtype]]) -> list:
    """Prepare input tensors based on the function signature."""
    inputs = []
    for arg in signature:
        inputs.append(torch.randn(arg[0], dtype=arg[1]))
    return inputs


def run_pytorch_function(func: Callable, inputs: list) -> torch.Tensor:
    """Run the PyTorch function with the given inputs."""
    with torch.no_grad():
        return func(*inputs)


def run_pytorch_file(input_file: str) -> torch.Tensor:
    print(f"Testing transpilation of {input_file}")

    function_name, signature = load_function_signature(input_file)
    pytorch_func = load_pytorch_function(input_file, function_name)

    inputs = prepare_inputs(signature)
    pytorch_output = run_pytorch_function(pytorch_func, inputs)

    return pytorch_output
