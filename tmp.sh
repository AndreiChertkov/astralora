#! /bin/bash
python script.py --task vgg19_tiny --name bb_all_r_10_s_1000 --mode bb --rank 10 --samples_bb 1000 --samples_sm 1000 --rewrite
python script.py --task vgg19_tiny --name bb_all_r_10_s_100 --mode bb --rank 10 --samples_bb 100 --samples_sm 100 --rewrite
python script.py --task vgg19_tiny --name bb_all_r_1_s_1000 --mode bb --rank 1 --samples_bb 1000 --samples_sm 1000 --rewrite
python script.py --task vgg19_tiny --name bb_all_r_100_s_1000 --mode bb --rank 100 --samples_bb 1000 --samples_sm 1000 --rewrite
python script.py --task vgg19_tiny --name bb_all_r_100_s_100 --mode bb --rank 100 --samples_bb 100 --samples_sm 100 --rewrite
