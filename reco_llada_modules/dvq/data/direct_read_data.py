from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import os
import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from glob import glob
import pyarrow.parquet as pq
from pyarrow.fs import FileSystem, FileSelector, SubTreeFileSystem

def read_embs(files):
    features = []
    pids = []
    for fn in tqdm(files):
        ckpt = torch.load(fn, weights_only=True, map_location="cpu")
        features.append(ckpt['features'])
        pids.append(ckpt['pids'])
    embs = torch.cat(features, dim=0)
    pids = torch.cat(pids, dim=0)
    return embs, pids

class UserSeqDataset(IterableDataset):

    def __init__(self, emb_dir, hive_file_list):
        files = sorted(glob(os.path.join(emb_dir, "*.pth")))
        embs, pids = read_embs(files)
        self.embs = embs
        self.pids = pids
        self.pid2index = {pid: i for i, pid in enumerate(pids.numpy())}
        with open(hive_file_list, "r") as f:
            self.files_list = [x.strip() for x in f.readlines() if x.strip() != '']
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_list = self.files_list
        np.random.shuffle(file_list)
        MASK = ((1 << 48) - 1) 
        for fn in file_list:
            print("read fn", fn)
            df = pq.read_table(fn, columns=['246', 'user_id']).to_pandas()
            for uid, seq in zip(df['user_id'], df['246'].values):
                for pid in seq:
                    pid = int(pid) & MASK
                    if pid in self.pid2index:
                        yield self.embs[self.pid2index[pid]], (pid, uid)


class UserSeqData(pl.LightningDataModule):

    def __init__(self, emb_dir, hive_file_list, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.dataloader = None
    
    def train_dataloader(self):
        if self.dataloader is None:
            dataset = UserSeqDataset(self.hparams.emb_dir, self.hparams.hive_file_list)
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                drop_last=True,
                pin_memory=True,
            )
        return self.dataloader
    
    def val_dataloader(self):
        return self.train_dataloader()
    
    def test_dataloader(self):
        return self.train_dataloader()


class EmbeddingData(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        files = sorted(glob(os.path.join(data_dir, "*.pth")))
        files = files
        num_train_files = int(len(files) * 0.8)
        self.infer_files = files
        self.train_files = files[:num_train_files]
        self.val_files = files[num_train_files:]
    
    def train_dataloader(self):
        r = read_embs(self.train_files)
        dataset = TensorDataset(*r)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader
    
    def val_dataloader(self):
        r = read_embs(self.val_files)
        dataset = TensorDataset(*r)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
    
    def test_dataloader(self):
        r = read_embs(self.infer_files)
        dataset = TensorDataset(*r)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader


class ParquetData(pl.LightningDataModule):
    
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        files = sorted(glob(os.path.join(data_dir, "*.pth")))
        fs, path = FileSystem.from_uri(data_dir)
        ds = pq.ParquetDataset(path, fs)
        self.num_files = 70#len(ds.fragments)
        self.num_train_files = 60# int(self.num_files * 0.8)
        self.ds = ds

    def read_parquet(self, begin, end):
        features = []
        pids = []
        for i in range(begin, end):
            fragment = self.ds.fragments[i]
            df = fragment.to_table(
                columns=["photo_id", "embedding"]
            ).to_pandas()
            features.append(np.stack(df['embedding']))
            pids.append(df['photo_id'])
        embs = np.concatenate(features, axis=0)
        embs = torch.tensor(embs)
        pids = np.concatenate(pids, axis=0)
        pids = torch.tensor(pids)
        return embs, pids
    
    def train_dataloader(self):
        r = self.read_parquet(begin=0, end=self.num_train_files)
        dataset = TensorDataset(*r)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader
    
    def val_dataloader(self):
        r = self.read_parquet(begin=self.num_train_files, end=self.num_files)
        dataset = TensorDataset(*r)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
    
    def test_dataloader(self):
        r = self.read_parquet(begin=60, end=len(self.ds.fragments))
        dataset = TensorDataset(*r)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def demo_dataloader(self):
        r = self.read_parquet(begin=70, end=71)
        dataset = TensorDataset(*r)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader