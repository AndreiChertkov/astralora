#! /bin/bash
# python script.py --task vgg19_tiny --name bb_one2_r_10_s_1000 --mode bb --rank 10 --samples_bb 1000 --samples_sm 1000 --rewrite
# python script.py --task vgg19_tiny --name bb_one2_r_10_s_100 --mode bb --rank 10 --samples_bb 100 --samples_sm 100 --rewrite
# python script.py --task vgg19_tiny --name bb_one2_r_1_s_1000 --mode bb --rank 1 --samples_bb 1000 --samples_sm 1000 --rewrite
# python script.py --task vgg19_tiny --name bb_one2_r_100_s_1000 --mode bb --rank 100 --samples_bb 1000 --samples_sm 1000 --rewrite
# python script.py --task vgg19_tiny --name bb_one2_r_100_s_100 --mode bb --rank 100 --samples_bb 100 --samples_sm 100 --rewrite


# python script.py --task vgg19_tiny --name bb_one2_wres_r_100_sbb_-1_sm_100 --mode bb --rank 100 --samples_bb -1 --samples_sm 100 --use_residual


python script.py --name vitbb_r_10_s_1000 --task vgg19_tiny --arch vit_b_32 --epochs 300 --batch-size 256 --lr 1e-3 --weight-decay 0.1 --workers 4 --print-freq 50 --mode bb --rank 10 --samples_bb 1000 --samples_sm 1000 --replace-layers 11
python script.py --name vitbb_r_10_s_100 --task vgg19_tiny --arch vit_b_32 --epochs 300 --batch-size 256 --lr 1e-3 --weight-decay 0.1 --workers 4 --print-freq 50 --mode bb --rank 10 --samples_bb 100 --samples_sm 100 --replace-layers 11
python script.py --name vitbb_r_10_bbs_-1_sm_1000 --task vgg19_tiny --arch vit_b_32 --epochs 300 --batch-size 256 --lr 1e-3 --weight-decay 0.1 --workers 4 --print-freq 50 --mode bb --rank 10 --samples_bb -1 --samples_sm 1000 --replace-layers 11

python script.py --name vitbb_r_100_s_1000 --task vgg19_tiny --arch vit_b_32 --epochs 300 --batch-size 256 --lr 1e-3 --weight-decay 0.1 --workers 4 --print-freq 50 --mode bb --rank 100 --samples_bb 1000 --samples_sm 1000 --replace-layers 11
python script.py --name vitbb_r_100_s_100 --task vgg19_tiny --arch vit_b_32 --epochs 300 --batch-size 256 --lr 1e-3 --weight-decay 0.1 --workers 4 --print-freq 50 --mode bb --rank 100 --samples_bb 100 --samples_sm 100 --replace-layers 11
python script.py --name vitbb_r_100_bbs_-1_sm_1000 --task vgg19_tiny --arch vit_b_32 --epochs 300 --batch-size 256 --lr 1e-3 --weight-decay 0.1 --workers 4 --print-freq 50 --mode bb --rank 100 --samples_bb -1 --samples_sm 1000 --replace-layers 11

