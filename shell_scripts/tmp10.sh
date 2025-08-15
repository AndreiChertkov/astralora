# digital

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode digital --name digital_seed1 --seed 1


# baseline

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_baseline_gd-gd_seed1 --samples_bb -1 --skip_sm --seed 1


# without rebuilds

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank1_seed1 --samples_bb 1000 --samples_sm 1000 --rank 1 --seed 1

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank10_seed1 --samples_bb 1000 --samples_sm 1000 --rank 10 --seed 1

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank100_seed1 --samples_bb 1000 --samples_sm 1000 --rank 100 --seed 1


# --step_sm_rebuild 100 

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank1_rebuild100_seed1 --samples_bb 1000 --samples_sm 1000 --rank 1 --seed 1 --step_sm_rebuild 100 

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank10_rebuild100_seed1 --samples_bb 1000 --samples_sm 1000 --rank 10 --seed 1 --step_sm_rebuild 100 

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank100_rebuild100_seed1 --samples_bb 1000 --samples_sm 1000 --rank 100 --seed 1 --step_sm_rebuild 100 


# --step_sm_rebuild 10

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank1_rebuild10_seed1 --samples_bb 1000 --samples_sm 1000 --rank 1 --seed 1 --step_sm_rebuild 10

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank10_rebuild10_seed1 --samples_bb 1000 --samples_sm 1000 --rank 10 --seed 1 --step_sm_rebuild 10

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank100_rebuild10_seed1 --samples_bb 1000 --samples_sm 1000 --rank 100 --seed 1 --step_sm_rebuild 10


# --step_sm_rebuild 5

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank1_rebuild5_seed1 --samples_bb 1000 --samples_sm 1000 --rank 1 --seed 1 --step_sm_rebuild 5

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank10_rebuild5_seed1 --samples_bb 1000 --samples_sm 1000 --rank 10 --seed 1 --step_sm_rebuild 5

python script.py --task airbench_cifar --root airbench_cifar/result_mzi --mode bb --bb_kind mzi --name bb_mzi_rank100_rebuild5_seed1 --samples_bb 1000 --samples_sm 1000 --rank 100 --seed 1 --step_sm_rebuild 5