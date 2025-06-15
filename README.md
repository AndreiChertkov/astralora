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

4. TODO `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 3 --mode digital --name digital`

5. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 4 --mode bb --name bb_rank10_samples1 --rank 10 --samples_bb 1 --samples_sm 1`

6. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 6 --mode bb --name bb_rank10_samples10 --rank 10 --samples_bb 10 --samples_sm 10`

7. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 7 --mode bb --name bb_rank10_samples100 --rank 10 --samples_bb 100 --samples_sm 100`

8.

9. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 4 --mode bb --name b_stoch_samples1 --use_stochastic_w --samples_bb 1 --samples_sm 1`

10. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 6 --mode bb --name bb_stoch_samples10 --use_stochastic_w --samples_bb 10 --samples_sm 10`

11. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 7 --mode bb --name bb_stoch_samples100 --use_stochastic_w --samples_bb 100 --samples_sm 100`

12.

### OLD_RUNS

5. `clear && torchrun --standalone --nproc_per_node=2 run.py --gpus 2,3 --mode bb --name bb_rank10 --rank 10`

6. `clear && torchrun --standalone --nproc_per_node=2 run.py --gpus 4,5 --mode bb --name bb_rank100 --rank 100`

7. `clear && torchrun --standalone --nproc_per_node=2 run.py --gpus 6,7 --mode bb --name bb_stoch --use_stochastic_w --batch_size 2 --samples_bb 50 --samples_sm 50`

8. `clear && torchrun --standalone --nproc_per_node=2 run.py --gpus 6,7 --mode bb --name bb_rank1 --rank 1`

#### todo

- `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 4 --mode bb --name bb_rank10_samples250 --rank 10 --samples_bb 250 --samples_sm 250`



- `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 7 --mode bb --name bb_stoch_samples250 --use_stochastic_w --samples_bb 250 --samples_sm 250  --batch_size 2`