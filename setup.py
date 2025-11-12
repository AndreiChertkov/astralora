from setuptools import find_packages
from setuptools import setup


setup(
    name="astralora",
    version="0.2",
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
        "torch>=2.1.0",
        "torchvision>=0.15.0",
        "torchaudio==2.8.0",
        "tqdm",
        "transformers",
        "tensorboard",
        "scikit-learn",
        "speechbrain",
        "pillow",
    ],
) 