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

4. Run the computation with the command like `torchrun --standalone --nproc_per_node=1 run.py --gpus 3 --mode digital --name digital`


## Computations

### nanogpt_fineweb (hopper)

1. TODO `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 3 --mode digital --name digital`

2. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 4 --mode bb --name bb_rank10_samples1 --rank 10 --samples_bb 1 --samples_sm 1`

3. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 6 --mode bb --name bb_rank10_samples10 --rank 10 --samples_bb 10 --samples_sm 10`

4. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 7 --mode bb --name bb_rank10_samples100 --rank 10 --samples_bb 100 --samples_sm 100`

5.

6. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 4 --mode bb --name b_stoch_samples1 --use_stochastic_w --samples_bb 1 --samples_sm 1`

7. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 6 --mode bb --name bb_stoch_samples10 --use_stochastic_w --samples_bb 10 --samples_sm 10`

8. `clear && torchrun --standalone --nproc_per_node=1 run.py --gpus 7 --mode bb --name bb_stoch_samples100 --use_stochastic_w --samples_bb 100 --samples_sm 100`

9.

### nanogpt_fineweb (jobs)

- `clear && python script.py --mode digital --name digital`

- `clear && python script.py --mode bb --name bb_rank10_samples1 --rank 10 --samples_bb 1 --samples_sm 1`

- `clear && python script.py --mode bb --name bb_rank10_samples10 --rank 10 --samples_bb 10 --samples_sm 10`

- `clear && python script.py --mode bb --name bb_rank10_samples100 --rank 10 --samples_bb 100 --samples_sm 100`

- `clear && python script.py --mode bb --name bb_rank10_samples500 --rank 10 --samples_bb 500 --samples_sm 500`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples1 --rank 10 --samples_bb 1 --samples_sm 1`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples10 --rank 10 --samples_bb 10 --samples_sm 10`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples100 --rank 10 --samples_bb 100 --samples_sm 100`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples500 --rank 10 --samples_bb 500 --samples_sm 500`

- `clear && python script.py --mode bb --name bb_rank1_samples100 --rank 1 --samples_bb 100 --samples_sm 100`

- `clear && python script.py --mode bb --name bb_rank100_samples100 --rank 100 --samples_bb 100 --samples_sm 100`

- `clear && python script.py --mode digital --name digital_seed2 --seed 2`

- `clear && python script.py --mode digital --name digital_seed3 --seed 3`

- `clear && python script.py --mode bb --name bb_rank10_samples100_seed2 --rank 10 --samples_bb 100 --samples_sm 100 --seed 2`

- `clear && python script.py --mode bb --name bb_rank10_samples100_seed3 --rank 10 --samples_bb 100 --samples_sm 100 --seed 3`