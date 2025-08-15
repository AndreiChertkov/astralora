python script.py --task airbench_cifar --root debug_mzi4 --mode bb --bb_kind mzi --name bb-gd_sm-rank100 --samples_bb -1 --samples_sm 1000 --rank 100 --step_sm_rebuild 100

python script.py --task airbench_cifar --root debug_mzi4 --mode bb --bb_kind mzi --name bb-stoch_sm-rank100 --samples_bb 1000 --samples_sm 1000 --rank 100 --step_sm_rebuild 100