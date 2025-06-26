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


clear && torchrun --standalone --nproc_per_node=1 nanogpt_fineweb/run.py --gpus 7 --mode bb --name bb_stoch_samples100_test --use_stochastic_w --samples_bb 100 --samples_sm 100 --rewrite --root_data ./nanogpt_fineweb/_data/fineweb --root ./nanogpt_fineweb/result


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
    > RUNS `cryri --logs 699dd35e`

- `clear && python script.py --mode bb --name bb_rank10_samples1 --rank 10 --samples_bb 1 --samples_sm 1`
    > RUNS `cryri --logs d416bc1f`

- `clear && python script.py --mode bb --name bb_rank10_samples10 --rank 10 --samples_bb 10 --samples_sm 10`
    > RUNS `cryri --logs 1d3cfbd1`

- `clear && python script.py --mode bb --name bb_rank10_samples100 --rank 10 --samples_bb 100 --samples_sm 100`
    > RUNS `cryri --logs 2f74fe47`

- `clear && python script.py --mode bb --name bb_rank10_samples500 --rank 10 --samples_bb 500 --samples_sm 500`
    > RUNS `cryri --logs 4bf8cee9`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples1 --rank 10 --samples_bb 1 --samples_sm 1`
    > RUNS `cryri --logs a63f3580`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples10 --rank 10 --samples_bb 10 --samples_sm 10`
    > RUNS `cryri --logs f8a2f856`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples100 --rank 10 --samples_bb 100 --samples_sm 100`
    > RUNS `cryri --logs bd3ff37e`

- `clear && python script.py --mode bb --use_stochastic_w --name bb_stoch_samples500 --rank 10 --samples_bb 500 --samples_sm 500`
    > RUNS `cryri --logs d98a8351`

- `clear && python script.py --mode bb --name bb_rank1_samples100 --rank 1 --samples_bb 100 --samples_sm 100`
    > RUNS `cryri --logs 964a1849`

- `clear && python script.py --mode bb --name bb_rank100_samples100 --rank 100 --samples_bb 100 --samples_sm 100`
    > RUNS `cryri --logs 3479d5e7`

- `clear && python script.py --mode digital --name digital_seed2 --seed 2`
    > RUNS `cryri --logs c5eb258a`

- `clear && python script.py --mode digital --name digital_seed3 --seed 3`
    > RUNS `cryri --logs e12349b8`

## new run

- `clear && python script.py --mode digital --name digital --root result_new`
    > DONE `cryri --logs 58716ad6`

- `clear && python script.py --mode bb --name bb_rank10_gd_update_iters1_lr4 --rank 10 --root result_new --use_gd_update --gd_update_iters 1`
    > DONE `cryri --logs f4b44ae1`

 - `clear && python script.py --mode bb --name bb_rank10_gd_update_iters10_lr4 --rank 10 --root result_new --use_gd_update --gd_update_iters 10`
    > DONE `cryri --logs cff4be3a`

 - `clear && python script.py --mode bb --name bb_rank10_gd_update_iters100_lr4 --rank 10 --root result_new --use_gd_update --gd_update_iters 100`
    > DONE `cryri --logs 0bf5cd38`

 - `clear && python script.py --mode bb --name bb_rank10_gd_update_iters1000_lr4 --rank 10 --root result_new --use_gd_update --gd_update_iters 1000`
    > DONE `cryri --logs 235629bd`


## new run 2

- `clear && python script.py --mode digital --name digital --root result_new2`
    > PEND `cryri --logs ef69976d`

- `clear && python script.py --mode bb --name bb_rank10_samples100 --rank 10 --samples_bb 100 --samples_sm 100 --root result_new2`
    > PEND `cryri --logs 62f655d8`

- `clear && python script.py --mode bb --name bb_slm_rank10_samples100 --rank 10 --samples_bb 100 --samples_sm 100 --root result_new2 --bb_kind slm`
    > PEND `cryri --logs 5d2020ce`

    