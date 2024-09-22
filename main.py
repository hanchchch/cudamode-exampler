import logging
import os
import sys

from src.example_models import models
from src.generators.python_function import write_functions
from src.generators.python_signature import write_signature
from src.linter import is_valid_python

output_dir = "outputs"
output_md_dir = "outputs/md"
output_py_dir = "outputs/py"


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    i = 0
    for model in models.keys():
        i += 1
        logger.info(f"--- {i}/{len(models)}: Processing {model}")
        py_filepaths = write_functions(
            reference_model=model,
            output_md_dir=output_md_dir,
            output_py_dir=output_py_dir,
        )
        for filepath in py_filepaths:
            write_signature(filepath)
            if not is_valid_python([filepath]):
                logger.warning(f"Invalid python code: {filepath}")
                os.remove(filepath)
                continue
        break
