"""Download the GPT-2 tokens of Fineweb from the huggingface

See https://huggingface.co/datasets/HuggingFaceFW/fineweb for details.

We load prepared data, and it saves about an hour of startup time compared
to regenerating them. Run this script to download 2.67B Fineweb tokens with the
specified number of downloaded chunks, e.g.:
"python run_data.py --data_chunks 27".

"""
from huggingface_hub import hf_hub_download
import os


from config import config


def data_get(fname, root):
    fold = os.path.join(os.path.dirname(__file__), root)
    if not os.path.exists(os.path.join(fold, fname)):
        hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', repo_type='dataset',
            local_dir=fold, filename=fname)
    else:
        print(f'--- Skip: already downloaded ("{fname}")')


if __name__ == '__main__':
    args = config()

    if args.data_chunks < 1 or args.data_chunks > 103:
        raise ValueError('Invalid value for "args.data_chunks"')

    for i in range(1, args.data_chunks+1):
        data_get('fineweb_train_%06d.bin'%i, args.root_data)

    data_get("fineweb_val_%06d.bin"%0, args.root_data)