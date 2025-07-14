cd ..

# experiments for ecapa_urbansound8k
# check what happens for epoch = 50 :)

python script.py --task ecapa_urbansound8k --name tmp1 --mode digital --epochs 5

python script.py --task ecapa_urbansound8k --name tmp2 --mode digital --epochs 50

python script.py --task ecapa_urbansound8k --name tmp3 --mode digital --epochs 51

python script.py --task ecapa_urbansound8k --name tmp4 --mode digital --epochs 100