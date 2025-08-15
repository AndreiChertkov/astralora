python script.py --task airbench_cifar --root debug_mzi3 --mode bb --bb_kind mzi --name bb-gd_sm-gd --samples_bb -1 --skip_sm

python script.py --task airbench_cifar --root debug_mzi3 --mode bb --bb_kind mzi --name bb-gd_sm-svd --samples_bb -1 --samples_sm -1

python script.py --task airbench_cifar --root debug_mzi3 --mode bb --bb_kind mzi --name bb-gd_sm-rank100 --samples_bb -1 --samples_sm 1000 --rank 100

python script.py --task airbench_cifar --root debug_mzi3 --mode bb --bb_kind mzi --name bb-stoch_sm-gd --samples_bb 1000 --skip_sm

python script.py --task airbench_cifar --root debug_mzi3 --mode bb --bb_kind mzi --name bb-stoch_sm-svd --samples_bb 1000 --samples_sm -1

python script.py --task airbench_cifar --root debug_mzi3 --mode bb --bb_kind mzi --name bb-stoch_sm-rank100 --samples_bb 1000 --samples_sm 1000 --rank 100