# -----------------------------------------------------------------------------
# Digital

python script.py --task airbench_cifar --rewrite --mode digital --name digital 

python script.py --task cnn_cifar --rewrite --mode digital --name digital

python script.py --task nanogpt_fineweb --rewrite --mode digital --name digital --torchrun 1

python script.py --task ecapa_urbansound8k --rewrite --mode digital --name digital 


# -----------------------------------------------------------------------------
# Baseline gd-full for matvec

python script.py --task airbench_cifar --rewrite --mode bb --name baseline_full_matvec --bb_kind matvec --samples_bb -1 --skip_sm

python script.py --task cnn_cifar --rewrite --mode bb --name baseline_full_matvec  --bb_kind matvec --samples_bb -1 --skip_sm

python script.py --task nanogpt_fineweb --rewrite --mode bb --name baseline_full_matvec --bb_kind matvec --torchrun 1 --samples_bb -1 --skip_sm

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name baseline_full_matvec --bb_kind matvec --samples_bb -1 --skip_sm


# -----------------------------------------------------------------------------
# Baseline gd-full for slm

python script.py --task airbench_cifar --rewrite --mode bb --name baseline_full_slm --bb_kind slm --samples_bb -1 --skip_sm

python script.py --task cnn_cifar --rewrite --mode bb --name baseline_full_slm  --bb_kind slm --samples_bb -1 --skip_sm

python script.py --task nanogpt_fineweb --rewrite --mode bb --name baseline_full_slm --bb_kind slm --torchrun 1 --samples_bb -1 --skip_sm

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name baseline_full_slm --bb_kind slm --samples_bb -1 --skip_sm


# -----------------------------------------------------------------------------
# Baseline gd-full for mrr

python script.py --task airbench_cifar --rewrite --mode bb --name baseline_full_mrr --bb_kind mrr --samples_bb -1 --skip_sm

python script.py --task cnn_cifar --rewrite --mode bb --name baseline_full_mrr  --bb_kind mrr --samples_bb -1 --skip_sm

python script.py --task nanogpt_fineweb --rewrite --mode bb --name baseline_full_mrr --bb_kind mrr --torchrun 1 --samples_bb -1 --skip_sm

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name baseline_full_mrr --bb_kind mrr --samples_bb -1 --skip_sm


# -----------------------------------------------------------------------------
# Baseline for matvec (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name baseline_matvec_rank100 --bb_kind matvec --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name baseline_matvec_rank100  --bb_kind matvec --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name baseline_matvec_rank100 --bb_kind matvec --torchrun 1 --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name baseline_matvec_rank100 --bb_kind matvec --samples_bb -1 --samples_sm -1 --rank 100


# -----------------------------------------------------------------------------
# Baseline for slm (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name baseline_slm_rank100 --bb_kind slm --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name baseline_slm_rank100  --bb_kind slm --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name baseline_slm_rank100 --bb_kind slm --torchrun 1 --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name baseline_slm_rank100 --bb_kind slm --samples_bb -1 --samples_sm -1 --rank 100


# -----------------------------------------------------------------------------
# Baseline for mrr (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name baseline_mrr_rank100 --bb_kind mrr --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name baseline_mrr_rank100  --bb_kind mrr --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name baseline_mrr_rank100 --bb_kind mrr --torchrun 1 --samples_bb -1 --samples_sm -1 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name baseline_mrr_rank100 --bb_kind mrr --samples_bb -1 --samples_sm -1 --rank 100


# -----------------------------------------------------------------------------
# Computation for matvec with bb-gd (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name bb-gd_matvec_rank100 --bb_kind matvec --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name bb-gd_matvec_rank100  --bb_kind matvec --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name bb-gd_matvec_rank100 --bb_kind matvec --torchrun 1 --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name bb-gd_matvec_rank100 --bb_kind matvec --samples_bb -1 --samples_sm 1000 --rank 100


# -----------------------------------------------------------------------------
# Computation for slm with bb-gd (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name bb-gd_slm_rank100 --bb_kind slm --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name bb-gd_slm_rank100  --bb_kind slm --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name bb-gd_slm_rank100 --bb_kind slm --torchrun 1 --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name bb-gd_slm_rank100 --bb_kind slm --samples_bb -1 --samples_sm 1000 --rank 100


# -----------------------------------------------------------------------------
# Computation for mrr with bb-gd (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name bb-gd_mrr_rank100 --bb_kind mrr --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name bb-gd_mrr_rank100  --bb_kind mrr --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name bb-gd_mrr_rank100 --bb_kind mrr --torchrun 1 --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name bb-gd_mrr_rank100 --bb_kind mrr --samples_bb -1 --samples_sm 1000 --rank 100


# -----------------------------------------------------------------------------
# Computation for matvec (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name bb_matvec_rank100 --bb_kind matvec --samples_bb 1000 --samples_sm 1000 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name bb_matvec_rank100  --bb_kind matvec --samples_bb 1000 --samples_sm 1000 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name bb_matvec_rank100 --bb_kind matvec --torchrun 1 --samples_bb 100 --samples_sm 1000 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name bb_matvec_rank100 --bb_kind matvec --samples_bb 1000 --samples_sm 1000 --rank 100


# -----------------------------------------------------------------------------
# Computation for slm (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name bb_slm_rank100 --bb_kind slm --samples_bb 1000 --samples_sm 1000 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name bb_slm_rank100  --bb_kind slm --samples_bb 1000 --samples_sm 1000 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name bb_slm_rank100 --bb_kind slm --torchrun 1 --samples_bb 100 --samples_sm 1000 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name bb_slm_rank100 --bb_kind slm --samples_bb 1000 --samples_sm 1000 --rank 100


# -----------------------------------------------------------------------------
# Computation for mrr (rank 100)

python script.py --task airbench_cifar --rewrite --mode bb --name bb_mrr_rank100 --bb_kind mrr --samples_bb 1000 --samples_sm 1000 --rank 100

python script.py --task cnn_cifar --rewrite --mode bb --name bb_mrr_rank100  --bb_kind mrr --samples_bb 1000 --samples_sm 1000 --rank 100

python script.py --task nanogpt_fineweb --rewrite --mode bb --name bb_mrr_rank100 --bb_kind mrr --torchrun 1 --samples_bb 100 --samples_sm 1000 --rank 100

python script.py --task ecapa_urbansound8k --rewrite --mode bb --name bb_mrr_rank100 --bb_kind mrr --samples_bb 1000 --samples_sm 1000 --rank 100