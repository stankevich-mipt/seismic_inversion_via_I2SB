# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import pickle
from torch.utils.data import Dataset


class MultiLMDBDataset(Dataset):

    def __init__(self, list_of_lmdb_databases):

        self.all_keys  = []
        self.databases = []
        self.map_ids_to_db = dict({})
        self.total_files = 0

        for i, entry in enumerate(list_of_lmdb_databases):

            self.databases.append(entry.lmdb_data)
            self.all_keys.extend(entry.local_files)

            for _ in entry.local_files:
                self.map_ids_to_db[self.total_files] = i
                self.total_files += 1

    def __getitem__(self, idx):
        
        db = self.databases[self.map_ids_to_db[idx]]
        data_path, model_path = self.all_keys[idx]
        key = data_path + model_path
        
        with db.begin(write=False, buffers=True) as txn:
            bytedata = txn.get(key.encode('ascii'))
        
        entry = pickle.loads(bytedata)

        return entry['model'], entry['data']


    def __len__(self):
        return self.total_files
