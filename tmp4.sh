# -----------------------------------------------------------------------------
# Computation for matvec (rank 1)

python script.py --use_residual --task airbench_cifar --rewrite --mode bb --name resi_bb_matvec_rank1 --bb_kind matvec --samples_bb 1000 --samples_sm 1000 --rank 1

python script.py --use_residual --task cnn_cifar --rewrite --mode bb --name resi_bb_matvec_rank1  --bb_kind matvec --samples_bb 1000 --samples_sm 1000 --rank 1

python script.py --use_residual --task nanogpt_fineweb --rewrite --mode bb --name resi_bb_matvec_rank1 --bb_kind matvec --torchrun 1 --samples_bb 100 --samples_sm 1000 --rank 1

python script.py --use_residual --task ecapa_urbansound8k --rewrite --mode bb --name resi_bb_matvec_rank1 --bb_kind matvec --samples_bb 1000 --samples_sm 1000 --rank 1


# -----------------------------------------------------------------------------
# Computation for slm (rank 1)

python script.py --use_residual --task airbench_cifar --rewrite --mode bb --name resi_bb_slm_rank1 --bb_kind slm --samples_bb 1000 --samples_sm 1000 --rank 1

python script.py --use_residual --task cnn_cifar --rewrite --mode bb --name resi_bb_slm_rank1  --bb_kind slm --samples_bb 1000 --samples_sm 1000 --rank 1

python script.py --use_residual --task nanogpt_fineweb --rewrite --mode bb --name resi_bb_slm_rank1 --bb_kind slm --torchrun 1 --samples_bb 100 --samples_sm 1000 --rank 1

python script.py --use_residual --task ecapa_urbansound8k --rewrite --mode bb --name resi_bb_slm_rank1 --bb_kind slm --samples_bb 1000 --samples_sm 1000 --rank 1


# -----------------------------------------------------------------------------
# Computation for mrr (rank 1)

python script.py --use_residual --task airbench_cifar --rewrite --mode bb --name resi_bb_mrr_rank1 --bb_kind mrr --samples_bb 1000 --samples_sm 1000 --rank 1

python script.py --use_residual --task cnn_cifar --rewrite --mode bb --name resi_bb_mrr_rank1  --bb_kind mrr --samples_bb 1000 --samples_sm 1000 --rank 1

python script.py --use_residual --task nanogpt_fineweb --rewrite --mode bb --name resi_bb_mrr_rank1 --bb_kind mrr --torchrun 1 --samples_bb 100 --samples_sm 1000 --rank 1

python script.py --use_residual --task ecapa_urbansound8k --rewrite --mode bb --name resi_bb_mrr_rank1 --bb_kind mrr --samples_bb 1000 --samples_sm 1000 --rank 1