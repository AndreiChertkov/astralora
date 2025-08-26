# Multi-GPU

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec --name bb_matvec_rank1 --mode bb --rank 1 --bb_kind matvec --bb_num 12 --samples_bb 1000 --samples_sm 1000 --accumulation_steps 5 --device_num 4 --device_total 4 --rewrite

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec --name bb_matvec_rank100 --mode bb --rank 100 --bb_kind matvec --bb_num 12 --samples_bb 1000 --samples_sm 1000 --accumulation_steps 5 --device_num 4 --device_total 4 --rewrite

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_matvec --name bb_matvec_rank500 --mode bb --rank 500 --bb_kind matvec --bb_num 12 --samples_bb 1000 --samples_sm 1000 --accumulation_steps 5 --device_num 4 --device_total 4 --rewrite