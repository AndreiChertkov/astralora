from setuptools import find_packages
from setuptools import setup


setup(
    name="astralora",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0",
        "transformers",
        "tiktoken",
        "datasets",
        "opt_einsum",
        "tqdm",
        "numpy>=1.26",
        "rotary_embedding_torch",
        "peft",
        "huggingface-hub",
        "neptune"
    ],
) 