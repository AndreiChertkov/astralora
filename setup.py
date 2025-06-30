from setuptools import find_packages
from setuptools import setup


setup(
    name="astralora",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "huggingface-hub",
        "matplotlib",
        "neptune",
        "numpy>=1.26",
        "opt_einsum",
        "peft",
        "rotary_embedding_torch",
        "tiktoken",
        "torch>=2.6.0",
        "torchvision",
        "tqdm",
        "transformers",
        "tensorboard",
        "scikit-learn",
        "speechbrain"
    ],
) 