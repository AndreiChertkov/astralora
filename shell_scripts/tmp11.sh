python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec --name bb_matvec_baseline_gd-gd --mode bb --bb_kind matvec --bb_num 12 --samples_bb -1 --skip_sm

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_monarch --name bb_monarch_baseline_gd-gd --mode bb --bb_kind monarch --bb_num 12 --samples_bb -1 --skip_sm

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_mrr --name bb_mrr_baseline_gd-gd --mode bb --bb_kind mrr --bb_num 12 --samples_bb -1 --skip_sm

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_slm --name bb_slm_baseline_gd-gd --mode bb --bb_kind slm --bb_num 12 --samples_bb -1 --skip_sm