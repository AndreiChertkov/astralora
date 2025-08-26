# Baseline for 3 layers

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec_3layers --name bb_matvec_rank1_bb-gd --mode bb --rank 1 --bb_kind matvec --bb_num 3 --samples_bb -1 --samples_sm 1000

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec_3layers --name bb_matvec_rank100_bb-gd --mode bb --rank 100 --bb_kind matvec --bb_num 3 --samples_bb -1 --samples_sm 1000

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec_3layers --name bb_matvec_rank500_bb-gd --mode bb --rank 500 --bb_kind matvec --bb_num 3 --samples_bb -1 --samples_sm 1000

# BB for 3 layers

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec_3layers --name bb_matvec_rank1 --mode bb --rank 1 --bb_kind matvec --bb_num 3 --samples_bb 1000 --samples_sm 1000

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec_3layers --name bb_matvec_rank100 --mode bb --rank 100 --bb_kind matvec --bb_num 3 --samples_bb 1000 --samples_sm 1000

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec_3layers --name bb_matvec_rank500 --mode bb --rank 500 --bb_kind matvec --bb_num 3 --samples_bb 1000 --samples_sm 1000