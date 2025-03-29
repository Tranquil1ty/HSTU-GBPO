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
from emb_client import EmbClient
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import random


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

def intersection_2d(tensor_1, target_tensor):
    matches = (tensor_1.unsqueeze(2) == target_tensor.unsqueeze(1)).cpu()
    result = torch.sum(matches, dim=2, dtype=torch.int)
    result = torch.sum(result, dim=1, dtype=torch.int)
    result = torch.mean(result, dtype=torch.float)
    return result

def calc_overlap(emb, mask_emb, k=100):
    emb = torch.tensor(emb).cuda() # [N, dim]
    mask_emb = torch.tensor(mask_emb).cuda() # [M, dim]
    
    dist = (
        emb.pow(2).sum(1, keepdim=True)
        - 2 * emb @ mask_emb.t()
        + mask_emb.pow(2).sum(1, keepdim=True).t()
    )
    _, ind = (-dist).topk(k, dim=1)
    return ind
    

def infer_loop(dl, model, btq_sender, writer=None, skip_steps=0):
    dfs = []
    file_name = "data.pkl"
    if False and os.path.exists(file_name):
        print("data.pkl found. Read data from cache.")
        df = pd.read_pickle(file_name)
    else:
        for step, batch in tqdm(enumerate(dl)):
            if step <= skip_steps:
                continue
            try:
                x, pid = batch
                x = x.cuda()
                # x = model.recon_loss.inmap(x)
                x_hat, latent_loss, ind = model(x)
                code = ind

                df = pd.DataFrame({
                    "pid": pid.cpu().numpy(), 
                    "code": list(code.cpu().numpy().astype(np.uint16)),
                    "emb": list(x.cpu().numpy()),
                    "emb_hat": list(x_hat.detach().cpu().numpy())
                })
                for i in range(16):
                    x_i = model.mask_decode(code, idx=[i])
                    df[f"emb_mask{i}th"] = list(x_i.detach().cpu().numpy())
                
                for i in range(1, 17):
                    mask_idx = random.sample(range(16), i)
                    x_mask = model.mask_decode(code, idx=mask_idx)
                    df[f"emb_mask{i}"] = list(x_mask.detach().cpu().numpy())
                dfs.append(df)
                
            except Exception as e:
                print("error:",e)

        df = pd.concat(dfs)
        # df.to_pickle(file_name)

    unique_pids = df["pid"].nunique()
    df['code2'] = df['code'].apply(tuple)
    unique_codes = df["code2"].nunique()
    print("unique pid num:", unique_pids)
    print("unique code num:", unique_codes)
    print("unique ratio:", unique_codes/unique_pids)

    pid = np.array(df["pid"].tolist())
    emb = np.array(df["emb"].tolist())
    emb_hat = np.array(df["emb_hat"].tolist())

    begin=0
    end=1000
    emb_topk = calc_overlap(emb[begin:end, :], emb)
    pid_ia = pid[emb_topk.cpu().numpy()] #debug
    emb_hat_topk = calc_overlap(emb_hat[begin:end, :], emb)
    result = intersection_2d(emb_hat_topk, emb_topk)
    print("reconstruction: ", result)

    for i in range(16):
        emb_mask = np.array(df[f"emb_mask{i}th"].tolist())
        emb_mask_topk = calc_overlap(emb_mask[begin:end, :], emb)
        result = intersection_2d(emb_mask_topk, emb_topk)
        print(f"mask {i}_th token: ", result)

    for i in range(1,17):
        emb_mask = np.array(df[f"emb_mask{i}"].tolist())
        emb_mask_topk = calc_overlap(emb_mask[begin:end, :], emb)
        pid_mask = pid[emb_mask_topk.cpu().numpy()]
        result = intersection_2d(emb_mask_topk, emb_topk)
        print(f"mask {i} tokens: ", result)

    # for i in range(end):
    #     y = pid_ia[i].tolist()
    #     x = pid_mask[i].tolist()
    #     z = [i for i in x if i in y]
    #     z.sort()
    #     # print(y)
    #     # print(x)
    #     print(z)
    #     input()
    return 


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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt", type=str)

    parser.add_argument("--btq_prefix", type=str, default="hierarchical_semantic_id_v1_s")
    parser.add_argument("--shard_num", type=int, default=2)
    parser.add_argument("--kafka", type=str, default='hierarchical_semantic_id_v1_missed_key')
    # parser.add_argument("--ia_kess", type=str, default='grpc_mmu_e2e_i2i_PsCloud')
    parser.add_argument("--ia_kess", type=str, default='grpc_mmu_e2e_i2i_PsCloud')
    parser.add_argument("--ia_shard_num", type=int, default=32)
    parser.add_argument("--ia_biz", type=str, default="reco")
    
    args = parser.parse_args()
    args.ckpt='/pub/renlejian/repo/deep-vector-quantization/dvq/lightning_logs/version_18/checkpoints/epoch=38-step=241917.ckpt'
    args.batch_size=8192
    emb_cfg = dict(biz_df=args.ia_biz, grpc_service_name=args.ia_kess, shards=args.ia_shard_num)

    model = VQVAE.load_from_checkpoint(args.ckpt, args=args)
    model.eval()
    model = model.cuda()

    data_module = ParquetData(args.data_dir, args.batch_size, args.num_workers)

    infer_loader = data_module.demo_dataloader()

    btq_sender = None

    infer_loop(infer_loader, model, btq_sender)

if __name__ == '__main__':
    main()