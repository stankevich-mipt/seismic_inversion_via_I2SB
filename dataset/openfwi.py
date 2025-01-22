import os
import json 
import lmdb
import torch
import pickle
import numpy as np
import blobfile as bf

from cv2 import resize
from torch.utils.data import Dataset
from torchvision.transforms import Compose


_DATASET_KV_MAP = {
    "FlatVel_A": "flatvel-a",
    "FlatVel_B": "flatvel-b",
    "CurveVel_A": "curvevel-a",
    "CurveVel_B": "curvevel-b",
    "FlatFault_A": "flatfault-a",
    "FlatFault_B": "flatfault-b",
    "CurveFault_A": "curvefault-a",
    "CurveFault_B": "curvefault-b",
}

_WRITE_FREQUENCY = 200


def _list_npy_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_npy_files_recursively(full_path))
    return results


def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid


def log_transform(data, k=1, c=0):
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


class LogTransform(object):
    def __init__(self, k=1, c=0):
        self.k = k
        self.c = c

    def __call__(self, data):
        return log_transform(data, k=self.k, c=self.c)

class MinMaxNormalize(object):
    def __init__(self, datamin, datamax, scale=2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale

    def __call__(self, vid):
        return minmax_normalize(vid, self.datamin, self.datamax, self.scale)


class FWIDataset(Dataset):
   
    def __init__(
        self, data_files, model_files, lmdb_data=None, 
    ):
        
        self.local_files = list(zip(data_files, model_files)) 
        self.lmdb_data   = lmdb_data        

    def load(self, idx):
            
        data_path, model_path = self.local_files[idx]

        data = np.load(data_path).astype('float32')
        model = np.load(model_path).astype('float32')        
        
        return data, model
        
    def lmdb_load(self, idx):
        
        data_path, model_path = self.local_files[idx]
        total_path = data_path + model_path

        with self.lmdb_data.begin(write=False, buffers=True) as txn:
            bytedata = txn.get(total_path.encode('ascii'))
        
        entry = pickle.loads(bytedata)

        return entry['model'], entry['data']

    def __getitem__(self, idx):

        if self.lmdb_data is not None:          
            return self.lmdb_load(idx)
        else:
            return self.load(idx)
        
    def __len__(self):

        return len(self.local_files)


def _build_lmdb_dataset(opt, root, ctx, log):
    
    root = str(root)
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(root, 'database.lmdb.pt')
    lmdb_path = os.path.join(root, 'database.lmdb')

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        log.info('[Dataset] Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    
    else:

        data_files = _list_npy_files_recursively(os.path.join(root, 'data'))
        model_files = _list_npy_files_recursively(os.path.join(root, 'model'))

        data_set = FWIDataset(data_files, model_files)

        torch.save(data_set, pt_path, pickle_protocol=4)
    
        log.info('[Dataset] Saving pt to {}'.format(pt_path))
        log.info('[Dataset] Building lmdb to {}'.format(lmdb_path))

        env = lmdb.open(lmdb_path, map_size=int(1e12))
        txn = env.begin(write=True)

        image_size = opt.image_size

        static_transform_data = Compose([
            LogTransform(),
            MinMaxNormalize(log_transform(ctx['data_min']), log_transform(ctx['data_max']))
        ])
        static_transform_model = Compose([
            MinMaxNormalize(ctx['label_min'], ctx['label_max'])
        ])
        
        for i, each in enumerate(data_set):
            
            data, model = each
            data = static_transform_data(data[None, ...]).squeeze(0)
            model = static_transform_model(model[None, ...]).squeeze(0)

            data = np.transpose(data, (1, 2, 0))
            data = resize(data, (image_size, image_size))
            data = np.transpose(data, (2, 0, 1))

            model = resize(model[0], (image_size, image_size))[None, ...]
            
            
            # use concatenation of file paths as a hash for lmdb entry
            data_path, model_path = data_set.local_files[i]
            total_path = data_path + model_path
            txn.put(total_path.encode('ascii'), pickle.dumps({'data': data, 'model': model}))

            if i % _WRITE_FREQUENCY == 0:
                txn.commit()
                txn = env.begin(write=True)
        
        txn.commit()
        
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False
    )

    return data_set

def build_lmdb_dataset(opt, log, train):
    """ resize -> crop -> to_tensor -> norm(-1,1) """
    split_dir = opt.dataset_dir / ('train' if train else 'val')

    with open(opt.json_data_config) as f: 
        ctx = json.load(f)[_DATASET_KV_MAP[opt.dataset_name]]

    dataset = _build_lmdb_dataset(opt, split_dir, ctx, log)
    log.info(f"[Dataset] Built Imagenet dataset {split_dir=}, size={len(dataset)}!")
    return dataset
