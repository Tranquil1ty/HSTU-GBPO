import os
import math
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import pandas as pd
import numpy as np

from proto.model_pb2 import ModelUpdateMessage, ModelItem
# from content_rq_vae.datasets.parquet import ParquetWriter
from framework_wrappers import BtqClient
from framework_wrappers import BtqException
from framework_wrappers import BtqErrorCode

from data.direct_read_data import EmbeddingData, UserSeqData, ParquetData
from data.kafka_fetch_data_clsdb_client import MissedKeyData
from vqvae import VQVAE

class BTQSender(object):

    def __init__(self, btq_prefix, shard_num):
        self.btq_prefix = btq_prefix
        self.btq_shard_num = shard_num
    
    def to_model_update_message(self, df, expire_timet=86400*90):
        msg = ModelUpdateMessage()
        msg.embedding_weight_size = -1
        for pid, code in zip(df.pid, df.code):
            item = ModelItem(embedding_weight=code.tobytes(), slot=0, sign=pid, expire_timet=expire_timet)
            msg.item.append(item)
        return msg
    
    def send(self, df):
        for s in range(self.btq_shard_num):
            cur = df[df.pid % self.btq_shard_num == s].copy()
            msg = self.to_model_update_message(cur)
            BtqClient.produce(f"{self.btq_prefix}{s}", [msg.SerializeToString()])

def infer_loop(dl, model, btq_sender, writer=None, skip_steps=0):
    all = 0
    valid = 0
    for step, batch in tqdm(enumerate(dl)):
        if step <= skip_steps:
            continue
        try:
            x, pid, all_num, valid_num = batch
            # print(x.shape)
            x = x.cuda()
            # x = model.recon_loss.inmap(x)
            x_hat, latent_loss, ind = model(x)
            code = ind#.squeeze(1)
            df = pd.DataFrame({"pid": pid.cpu().numpy(), "code": list(code.cpu().numpy().astype(np.uint16))})
            btq_sender.send(df)

            all += all_num
            valid += valid_num
            print(f"sample hit rate:  [{valid}/{all}]({valid * 1.0 / all}), [{valid_num}/{all_num}]({valid_num * 1.0 / all_num})" )
        except Exception as e:
            print("btq error:",e)


def main():

    parser = ArgumentParser()
    parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
    parser.add_argument("--enc_dec_flavor", type=str, default='deepmind', choices=['deepmind', 'openai'])
    parser.add_argument("--loss_flavor", type=str, default='l2', choices=['l2', 'logit_laplace'])
    parser.add_argument("--input_dim", type=int, default=128, help="input dim")
    parser.add_argument("--num_embeddings", type=int, default=512, help="vocabulary size; number of possible discrete states")
    parser.add_argument("--embedding_dim", type=int, default=32, help="size of the vector of the embedding of each discrete token")
    parser.add_argument("--n_hid", type=int, default=512, help="number of channels controlling the size of the model")
    parser.add_argument("--n_token", type=int, default=16, help="number of qunatized tokens")
    parser.add_argument("--data_dir", type=str, default='viewfs://hadoop-lt-cluster/home/reco_kaiworks/dw/reco_kaiworks.db/rlj_semantic_id_training_samples/p_date=20250311')
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt", type=str)

    parser.add_argument("--btq_prefix", type=str, default="hierarchical_semantic_id_v1_s")
    parser.add_argument("--shard_num", type=int, default=2)
    parser.add_argument("--kafka", type=str, default='hierarchical_semantic_id_v1_missed_key')
    # parser.add_argument("--kafka", type=str, default='imbalanced_semantic_id_missed_key')
    
    parser.add_argument("--ia_kess", type=str, default='grpc_mmu_e2e_i2i_PsCloud')
    parser.add_argument("--ia_shard_num", type=int, default=32)
    parser.add_argument("--ia_biz", type=str, default="reco")


    parser.add_argument("--ia_clsdb_kess", type=str, default='wxm_mm_sim_gsu_emb')
    parser.add_argument("--ia_clsdb_table", type=str, default='emb_wxm_mm_sim_gsu_128_new')
    parser.add_argument("--ia_data_type", type=str, default='float16')
    parser.add_argument("--ia_dim", type=int, default=128)
    
    args = parser.parse_args()
    args.ckpt='/pub/renlejian/repo/deep-vector-quantization/dvq/lightning_logs/version_18/checkpoints/epoch=38-step=241917.ckpt'
    # args.batch_size=None

    model = VQVAE.load_from_checkpoint(args.ckpt, args=args)
    model.eval()
    model = model.cuda()

    # data_module = EmbeddingData(args.data_dir, args.batch_size, args.num_workers)
    # data_module = ParquetData(args.data_dir, args.batch_size, args.num_workers)

    # emb_cfg = dict(biz_df=args.ia_biz, grpc_service_name=args.ia_kess, shards=args.ia_shard_num)
    # data_module = MissedKeyData(args.kafka, "rlj_kafka_consume", emb_cfg, 60, args.batch_size, args.num_workers)

    clsdb_cfg = dict(model=args.ia_clsdb_kess, table=args.ia_clsdb_table, dtype=args.ia_data_type, dim=args.ia_dim)
    data_module = MissedKeyData(args.kafka, "rlj_kafka_consume", clsdb_cfg, 60, None, args.num_workers)

    infer_loader = data_module.test_dataloader()

    btq_sender = BTQSender(args.btq_prefix, args.shard_num)
    # writer = ParquetWriter(**cfg['infer']['writer'])

    infer_loop(infer_loader, model, btq_sender)

if __name__ == '__main__':
    main()