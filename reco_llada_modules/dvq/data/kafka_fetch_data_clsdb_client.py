import os
import sys
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from queue import Queue
import time
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import pytorch_lightning as pl

from infra.kafka import (
    ConsumerParameter,
    KsKafkaConsumer,
    MessageContext,
    FinishConsumeException
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from proto.model_pb2 import ModelUpdateMessage, ModelItem
from clsdb_client import clsdb_client
import struct



class MissedKeyDataset(KsKafkaConsumer, IterableDataset):
    def __init__(
        self, 
        topic: str, 
        group_id: str, 
        clsdb_client_config: dict,
        max_lag_second=60,
    ):
        parameter = ConsumerParameter(topic, group_id, True)
        super().__init__(parameter)
        self.emb_client = clsdb_client.ClsdbClient()
        self.queue = Queue(maxsize=10)
        self.max_lag_second = max_lag_second
        dtype=clsdb_client_config["dtype"]
        if dtype == 'float16':
            succ = self.emb_client.set_client_param(clsdb_client_config["model"], clsdb_client_config["table"], 0, clsdb_client_config["dim"], timeout_ms=5000, max_signs_per_request=2000, is_raw_data=True, raw_data_type='uint16')
        else:
            succ = self.emb_client.set_client_param(clsdb_client_config["model"], clsdb_client_config["table"], 0, clsdb_client_config["dim"], timeout_ms=5000, max_signs_per_request=2000, is_raw_data=True, raw_data_type=dtype)
        self.dtype = dtype
        self.dim = clsdb_client_config["dim"]


    def consume(self, message: bytes, context: MessageContext):
        if context.timestamp / 1000 >= time.time() - self.max_lag_second:
            try:
                keys = np.frombuffer(message, dtype=np.uint64).tolist()
                keys = list(set(keys))
                if self.dtype == 'float16':
                    x = self.emb_client.fetch_int_embedding(keys)
                    bytes = b''.join(struct.pack('H', value) for value in x)
                    embedding = np.frombuffer(bytes, dtype=np.float16, count=self.dim * len(keys))
                    embedding = np.reshape(embedding, [-1, self.dim]).astype(np.float32)
                else:
                    x = self.emb_client.fetch_int_embedding(keys)
                    embedding = np.array(x, dtype=self.dtype) # TODO: type mapping
                    embedding = np.reshape(embedding, [-1, self.dim]).astype(np.float32)
                photo_id = np.array(keys, dtype=np.int64)
                valid = np.sum(embedding, axis=1) != 0
                all_num = len(embedding)
                embedding = embedding[valid]
                valid_num = len(embedding)
                photo_id = photo_id[valid]
                assert len(embedding) == len(photo_id), "Shapes of embedding and pid mismatch"
                out = [embedding, photo_id, all_num, valid_num]
                self.queue.put(out)
            except Exception as e:
                print(f"kafka error: {e}")

    def __iter__(self):
        self.start()
        while True:
            cur = self.queue.get()
            yield cur


class MissedKeyData(pl.LightningDataModule):

    def __init__(self, topic, group_id, emb_client_config, max_lag_second, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.dataloader = None
    
    def train_dataloader(self):
        if self.dataloader is None:
            dataset = MissedKeyDataset(self.hparams.topic, self.hparams.group_id, self.hparams.emb_client_config, self.hparams.max_lag_second)
            drop_last = False if self.hparams.batch_size is None else True
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                drop_last=drop_last,
                pin_memory=True,
            )
        return self.dataloader
    
    def val_dataloader(self):
        return self.train_dataloader()
    
    def test_dataloader(self):
        return self.train_dataloader()
