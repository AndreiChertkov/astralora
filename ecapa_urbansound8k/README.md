See: https://github.com/speechbrain/speechbrain/tree/c75ab5489431fd0a2a7d21160bc37677801cb506/recipes/UrbanSound8k

Donwload data: wget https://goo.gl/8hY5ER

Unzip data: `tar -xvzf 8hY5ER` (it will be `UrbanSound8K` folder)

Replace in `config.yaml` `data_folder: !PLACEHOLDER` to `data_folder: _data/UrbanSound8K`

Install: speechbrain pip install matplotlib tensorboard scikit-learn

Model: ECAPA-TDNN (see the paper `ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification`) [см. также https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/ECAPA_TDNN.py]

Run: clear && python train.py config.yaml