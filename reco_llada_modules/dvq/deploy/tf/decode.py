from __future__ import print_function
import numpy as np

import os
import sys
import logging
import argparse
import functools
import math
# from mio_tensorflow.variable import set_infer_with_tf_variable
# from moe import MoeLayer, ModelArgs
# set_infer_with_tf_variable(True)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="photo_predict")
parser.add_argument("--dryrun", dest="dryrun", const=True, default=False, nargs="?")
parser.add_argument("--with_kai", action="store_true")
parser.add_argument("--text", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--dest_dir", type=str, default="./")

args = parser.parse_args()
print(args)

if not args.dryrun and not args.with_kai:
    # monkey patch
    import mio_tensorflow.patch as mio_tensorflow_patch

    mio_tensorflow_patch.apply()

import tensorflow as tf
from mio_tensorflow.config import MioConfig

logging.basicConfig()
base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "base.yaml")

config = MioConfig.from_base_yaml(
    base_config,
    with_kai=args.with_kai,
    label_with_kv=True,
    clear_embeddings=True,
    clear_params=True,
    dryrun=args.dryrun,
)

def cast_16(x, use_fp16=True):
    if not use_fp16:
        return x
    if isinstance(x, list) or isinstance(x, tuple):
        return [tf.cast(a, tf.float16) for a in x]
    return tf.cast(x, tf.float16)

custom_dtype = tf.float16 if args.fp16 else tf.float32

def silu(x):
    return x * tf.sigmoid(x)

output = []
data = np.load("model_weights.npz")

print(data.files)

vocab_size = 512
token_num = 16
vocab_dim = 32

tokens = tf.cast(config.get_extra_param("tokens", size=token_num, dtype=tf.float32), dtype=tf.int64)
# print(data['quantizer.embed.weight'].shape)
table = tf.constant(data['quantizer.embed.weight'],dtype=custom_dtype)
tokens_one_hot = tf.one_hot(tokens, depth=vocab_size) #[bs, token_num, vocab_size]

z_q = tf.matmul(tokens_one_hot, table) #[bs, token_num, vocab_dim]
mask = tf.expand_dims(tf.cast(tokens != vocab_size, custom_dtype), -1) #[bs, token_num, 1]
z_q_masked = z_q * mask
z_q_masked = tf.reshape(z_q_masked, [-1, token_num * vocab_dim])

# decoder
# print(data['decoder.w1.weight'].shape)
# print(data['decoder.w2.weight'].shape)
# print(data['decoder.w3.weight'].shape)
w1 = tf.constant(np.transpose(data['decoder.w1.weight']), dtype=custom_dtype)
w2 = tf.constant(np.transpose(data['decoder.w2.weight']), dtype=custom_dtype)
w3 = tf.constant(np.transpose(data['decoder.w3.weight']), dtype=custom_dtype)

x = silu(tf.matmul(z_q_masked, w1)) * tf.matmul(z_q_masked, w3)
out = tf.matmul(x, w2)

targets = [
    ("embs", tf.cast(out, dtype=custom_dtype)),
    ("z", tf.cast(z_q_masked, dtype=custom_dtype)),
]

config.dump_predict_config(
    "./" + args.dest_dir,
    targets,
    input_type=3,
    dump_mode=args.mode,
    text=False
)
