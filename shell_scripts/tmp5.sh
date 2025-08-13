# Restrart old computation:

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_monarch --name bb_monarch_rank50_baseline_gd --mode bb --rank 50 --bb_kind monarch --bb_num 12 --samples_bb -1 --samples_sm 1000


# New runs:

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb --name bb_matvec_rank100_samplesbb10 --mode bb --rank 100 --bb_kind matvec --bb_num 12 --samples_bb 10 --samples_sm 1000

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb --name bb_monarch_rank100_samplesbb10 --mode bb --rank 100 --bb_kind monarch --bb_num 12 --samples_bb 10 --samples_sm 1000

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb --name bb_mrr_rank100_samplesbb10 --mode bb --rank 100 --bb_kind mrr --bb_num 12 --samples_bb 10 --samples_sm 1000

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb --name bb_slm_rank100_samplesbb10 --mode bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 10 --samples_sm 1000