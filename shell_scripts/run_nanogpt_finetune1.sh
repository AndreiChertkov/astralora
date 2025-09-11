python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_finetune --name sparse_bb_slm_digital --mode digital --load_digital nanogpt_fineweb/result_finetune/digital

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_finetune --name sparse_bb_slm_random --mode bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 100 --samples_sm 1000 --load_digital nanogpt_fineweb/result_finetune/digital

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_finetune --name sparse_bb_slm_p1 --mode sparse_bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 100 --samples_sm 1000 --load_digital nanogpt_fineweb/result_finetune/digital --sparse_top_p 0.1

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_finetune --name sparse_bb_slm_p2 --mode sparse_bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 100 --samples_sm 1000 --load_digital nanogpt_fineweb/result_finetune/digital --sparse_top_p 0.2

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_finetune --name sparse_bb_slm_p3 --mode sparse_bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 100 --samples_sm 1000 --load_digital nanogpt_fineweb/result_finetune/digital --sparse_top_p 0.3

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_finetune --name sparse_bb_slm_p4 --mode sparse_bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 100 --samples_sm 1000 --load_digital nanogpt_fineweb/result_finetune/digital --sparse_top_p 0.4

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_finetune --name sparse_bb_slm_p5 --mode sparse_bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 100 --samples_sm 1000 --load_digital nanogpt_fineweb/result_finetune/digital --sparse_top_p 0.5