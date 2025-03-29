import torch
import torch.nn.functional as F
from colossus.client import Client
from argparse import ArgumentParser
from data.embedding import EmbeddingData
from vqvae2 import VQVAE
import numpy as np
import IPython
from utils import define_long_view, define_effective_view

# python3 test_recall_1.py --batch_size 8192 --n_token 32 --num_embeddings 512 --data_dir /llm_reco_ssd/luoxinchen/dinov2/features/ia_emb_short1k_20241206 --ckpt lightning_logs/version_27/checkpoints/epoch\=9-step\=200000.ckpt 

parser = ArgumentParser()

parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
parser.add_argument("--enc_dec_flavor", type=str, default='deepmind', choices=['deepmind', 'openai'])
parser.add_argument("--loss_flavor", type=str, default='l2', choices=['l2', 'logit_laplace'])
parser.add_argument("--input_dim", type=int, default=512, help="input dim")
parser.add_argument("--num_embeddings", type=int, default=512, help="vocabulary size; number of possible discrete states")
parser.add_argument("--embedding_dim", type=int, default=64, help="size of the vector of the embedding of each discrete token")
parser.add_argument("--n_hid", type=int, default=512 * 4, help="number of channels controlling the size of the model")
parser.add_argument("--n_token", type=int, default=16, help="number of qunatized tokens")
parser.add_argument("--data_dir", type=str, default='/llm_reco_ssd/luoxinchen/dinov2/features/ia_emb_20241117')
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--ckpt", type=str)

args = parser.parse_args()


model = VQVAE.load_from_checkpoint(args.ckpt, args=args)

client = Client("grpc_colossusRecoSimItemV3")
result = client.query(1624755238, timeout=100)

print(result)

ckpt = torch.load("/tmp/out.pth")

pids = ckpt['pids']
tokens = ckpt['tokens']
embs = ckpt['embs']

pid2index = {
    pid: i
    for i, pid in enumerate(pids.numpy())
}

def emb_topk(emb, k):
    emb = F.normalize(emb, dim=-1)
    r = (emb @ embs.t()).topk(k=50)
    print('r', r)
    return pids[r.indices]

def test_recall(uid, k=100, p=None, use_effective_view=False, use_long_view=False):
    result = client.query(uid)
    if use_effective_view:
        result['effective_view'] = define_effective_view(result['duration'],result['play_time'])
        result = result[result['effective_view']]
    if use_long_view:
        result['long_view'] = define_long_view(result['duration'],result['play_time'])
        result = result[result['long_view']]


    valid_pids, valid_indices = [], []
    for pid in result.photo_id.values:
        if pid in pid2index:
            valid_pids.append(pid)
            valid_indices.append(pid2index[pid])
    u_tokens = tokens[valid_indices]
    num_tokens = u_tokens.shape[1]
    res = []
    for pos in range(num_tokens):
        cur = u_tokens[:, pos]
        unique_tokens, counts = cur.unique(return_counts=True)
        print('counts', counts.sort(descending=True))
        if p is None:
            probs = (counts / counts.sum()).cpu().numpy()
            res.append(np.random.choice(unique_tokens.cpu().numpy(), p=probs))
        else:
            count_values, s_ind = counts.sort(descending=True)
            print(pos, 'unique_tokens', unique_tokens[s_ind], 'counts', count_values)
            count_values = count_values[:p]
            s_ind = s_ind[:p]
            probs = (count_values / count_values.sum()).cpu().numpy()
            s = np.random.choice(s_ind.cpu().numpy(), p = probs)
            res.append(unique_tokens[s])
    
    u_emb = model.decode(torch.tensor(res).unsqueeze(0))
    return emb_topk(u_emb, k)

IPython.embed()

