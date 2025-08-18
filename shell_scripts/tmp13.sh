python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb_new --name bb_matvec_rank100_samplesbb1000 --mode bb --rank 100 --bb_kind matvec --bb_num 12 --samples_bb 1000 --samples_sm 1000 --samples_bb_batch_frac 0.1

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb_new --name bb_monarch_rank100_samplesbb1000 --mode bb --rank 100 --bb_kind monarch --bb_num 12 --samples_bb 1000 --samples_sm 1000 --samples_bb_batch_frac 0.1

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb_new --name bb_mrr_rank100_samplesbb1000 --mode bb --rank 100 --bb_kind mrr --bb_num 12 --samples_bb 1000 --samples_sm 1000 --samples_bb_batch_frac 0.1

python script.py --task nanogpt_fineweb --torchrun 1 --root nanogpt_fineweb/result_test_bb_new --name bb_slm_rank100_samplesbb1000 --mode bb --rank 100 --bb_kind slm --bb_num 12 --samples_bb 1000 --samples_sm 1000 --samples_bb_batch_frac 0.1