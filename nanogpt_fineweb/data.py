import glob
import numpy as np
import torch


class DistributedDataLoader:
    def __init__(self, args, process_rank, num_processes, vld=False):
        self.args = args
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.vld = vld

        self.B = args.batch_size
        self.T = args.sequence_length
        
        self.init()
        self.load()
        self.reset()

    def advance(self):
        # Advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def info(self):
        text = ''
        text += 'Dataloader ' + ('(vld) ' if self.vld else '(trn) ') + ': '
        text += f'total tokens: {self.ntok_total} '
        text += f'across {len(self.files)} files.'
        return text

    def init(self):
        if self.vld:
            filename_pattern = f'{self.args.root_data}/fineweb_val_*.bin'
        else:
            filename_pattern = f'{self.args.root_data}/fineweb_train_*.bin'
        
        self.files = sorted(glob.glob(filename_pattern))
        
        msg = f'Did not find files that match the pattern {filename_pattern}'
        assert len(self.files) > 0, msg

    def load(self):
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= self.num_processes * self.B * self.T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])


def _load_data_shard(filename):
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)

        msg = 'magic number mismatch in the data .bin file'
        assert header[0] == 20240520, msg
        
        msg = 'unsupported version'
        assert header[1] == 1, msg
        
        ntok = header[2] # number of tokens (claimed)
        
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    
    msg = 'number of tokens read does not match header?'
    assert len(tokens) == ntok, msg
    
    return tokens


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print('ERROR: magic number mismatch in the data .bin file!')
        exit(1)
    assert header[1] == 1, 'unsupported version'
    ntok = header[2] # number of tokens (claimed)
    return ntok