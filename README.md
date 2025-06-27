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
    pip install transformers torch>=2.6.0 torchvision tiktoken datasets opt_einsum tqdm numpy==1.26 rotary_embedding_torch peft huggingface-hub neptune
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

### cnn_cifar

- `clear && python script.py --task cnn_cifar --mode digital --name digital`
    > PEND `cryri --logs dbe886db`

- `clear && python script.py --task cnn_cifar --mode bb --name bb_matvec_rank1_samples100 --rank 1 --samples_bb 100 --samples_sm 100`
    > PEND `cryri --logs 20ec0907`

- `clear && python script.py --task cnn_cifar --mode bb --name bb_matvec_rank10_samples100 --rank 10 --samples_bb 100 --samples_sm 100`
    > PEND `cryri --logs e802634b`

- `clear && python script.py --task cnn_cifar --mode bb --name bb_matvec_rank100_samples100 --rank 100 --samples_bb 100 --samples_sm 100`
    > PEND `cryri --logs 38bee80c`

### nanogpt_fineweb

> Do `cd nanogpt_fineweb && python run_data.py` before the first run

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