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
from emb_client import EmbClient


class MissedKeyDataset(KsKafkaConsumer, IterableDataset):
    def __init__(
        self, 
        topic: str, 
        group_id: str, 
        emb_client_config: dict,
        max_lag_second=60,
    ):
        parameter = ConsumerParameter(topic, group_id, True)
        super().__init__(parameter)
        self.emb_client = EmbClient(**emb_client_config)
        self.queue = Queue(maxsize=10)
        self.max_lag_second = max_lag_second

    def consume(self, message: bytes, context: MessageContext):
        if context.timestamp / 1000 >= time.time() - self.max_lag_second:
            try:
                keys = np.frombuffer(message, dtype=np.uint64)
                emb_dict = self.emb_client.get_emb(keys)
                pid, embedding = zip(*emb_dict.items())
                photo_id = np.array(pid, dtype=np.int64)
                embedding = [np.frombuffer(x, dtype=np.float32) for x in embedding]
                embedding = np.stack(embedding)
                # out = {"photo_id": photo_id, "embedding": embedding}
                out = [embedding, photo_id]
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