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

4. Install python-package for isolation of conda-environment (optional):
    ```bash
    conda install -c conda-forge conda-ecosystem-user-package-isolation -y
    ```

5. Install dependencies:
    ```bash
    pip install transformers torch==2.6.0 tiktoken datasets opt_einsum tqdm numpy==1.26 rotary_embedding_torch peft huggingface-hub neptune
    ```
    > In the case of errors do `conda install gcc_linux-64 -y && conda install gxx_linux-64 -y && conda install -c conda-forge libstdcxx-ng -y`

6. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name astralora --all -y
    ```

7. If there is an import error:
    ```
    pip install -e .
    ```
    

## Usage

### nanogpt_fineweb

1. `cd nanogpt_fineweb`

2. `python run_data.py`

3. `chmod +x ../set_neptune_env.sh && ../set_neptune_env.sh`

4. `torchrun --standalone --nproc_per_node=4 run.py --gpus 0,1,2,3 --mode digital --name digital`

5. `torchrun --standalone --nproc_per_node=4 run.py --gpus 4,5,6,7 --mode bb_one --name bb_one_rank10 --rank 10`