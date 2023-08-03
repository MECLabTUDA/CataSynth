import pickle
from collections import namedtuple

import lmdb
import torch
from torch.utils.data import Dataset

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'phase_label', 'tool_label', 'filename'])


class LMDB_Dataset(Dataset):
    
    """ Dataset for latent model codes extracted by VQ-VAE2 """

    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), \
               torch.from_numpy(row.phase_label), torch.from_numpy(row.tool_label), row.filename
