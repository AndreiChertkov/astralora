# astralora


## Description

This package `astralora` (**A**daptive **S**urrogate **TRA**ining with **LO**w **RA**nk) enables efficient backpropagation through non-differentiable layers via dynamically trained low-rank surrogates and zero-order gradient approximation.


## Installation

1. Install [anaconda](https://www.anaconda.com) package manager with [python](https://www.python.org)

2. Create a virtual environment:
    ```bash
    conda create --name astralora python=3.10 -y
    ```

3. Activate the environment:
    ```bash
    conda activate astralora 
    ```

4. Optionally install python-package for isolation of conda-environment:
    ```bash
    conda install -c conda-forge conda-ecosystem-user-package-isolation -y
    ```

5. Install dependencies:
    ```bash
    pip install -e .
    ```
    > In the case of errors do `conda install gcc_linux-64 -y && conda install gxx_linux-64 -y && conda install -c conda-forge libstdcxx-ng -y`

6. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name astralora --all -y
    ```
    

## Usage

We consider the application of our approach `astralora` to different classes of problems (architecture + domain), located in folders: `airbench_cifar`, `cnn_cifar`, `ecapa_urbansound8k`, `nanogpt_fineweb`. Each of the designated folders contains a main script `run.py`, which provides details on setting up and running.

We also performed various runs of calculations on clusters using script `autorun.py`, `script.py`, and shell scripts `shell_scripts/tmp*.sh`.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Artem Basharin](https://github.com/a-wernon)


---


> ✭__🚂  The stars that you give to **astralora**, motivate us to develop faster and add new interesting features to the code 😃