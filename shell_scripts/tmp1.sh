cd ..

# experiments for ecapa_urbansound8k
# check what happens for epoch = 50 :)

python script.py --task ecapa_urbansound8k --name tmp1 --mode digital --epochs 5

python script.py --task ecapa_urbansound8k --name tmp2 --mode digital --epochs 50

python script.py --task ecapa_urbansound8k --name tmp3 --mode digital --epochs 51

python script.py --task ecapa_urbansound8k --name tmp3 --mode digital --epochs 100


# experiments for nanogpt_fineweb
# baseline

python script.py --task nanogpt_fineweb --name bb_matvec_rank100_baseline --mode bb --bb_kind matvec --rank 100 --samples_bb -1 --skip_sm --torchrun 1

python script.py --task nanogpt_fineweb --name bb_monarch_rank100_baseline --mode bb --bb_kind monarch --rank 100 --samples_bb -1 --skip_sm --torchrun 1

python script.py --task nanogpt_fineweb --name bb_mrr_rank100_baseline --mode bb --bb_kind mrr --rank 100 --samples_bb -1 --skip_sm --torchrun 1

python script.py --task nanogpt_fineweb --name bb_slm_rank100_baseline --mode bb --bb_kind slm --rank 100 --samples_bb -1 --skip_sm --torchrun 1


# experiments for nanogpt_fineweb
# baseline: gd + svd

python script.py --task nanogpt_fineweb --name bb_matvec_rank100_baseline_gd_and_svd --mode bb --bb_kind matvec --rank 100 --samples_bb -1 --samples_sm -1 --torchrun 1

python script.py --task nanogpt_fineweb --name bb_monarch_rank100_baselinee_gd_and_svd --mode bb --bb_kind monarch --rank 100 --samples_bb -1 --samples_sm -1 --torchrun 1

python script.py --task nanogpt_fineweb --name bb_mrr_rank100_baselinee_gd_and_svd --mode bb --bb_kind mrr --rank 100 --samples_bb -1 --samples_sm -1 --torchrun 1

python script.py --task nanogpt_fineweb --name bb_slm_rank100_baselinee_gd_and_svd --mode bb --bb_kind slm --rank 100 --samples_bb -1 --samples_sm -1 --torchrun 1