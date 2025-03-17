from __future__ import print_function
import numpy as np

import os
import sys
import logging
import argparse
import functools
import math
from mio_tensorflow.variable import set_infer_with_tf_variable
set_infer_with_tf_variable(True)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="user_predict")
parser.add_argument("--dryrun", dest="dryrun", const=True, default=False, nargs="?")
parser.add_argument("--with_kai", action="store_true")
parser.add_argument("--text", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--cross_att_kv_cache", action="store_true")
parser.add_argument("--self_att_kv_cache", action="store_true")
parser.add_argument("--pre_cut_num", type=int, default=1024)
parser.add_argument("--dest_dir", type=str, default="predictBeamSearch")

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

def mask(inputs, key_masks=None, type=None):
    padding_num = -2 ** 32 + 1
    if type in ("key"):
        key_masks = tf.to_float(key_masks) # (N, seqlen)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
        print("--------key-----------")
    if type in ("future"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)
        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        print("-------future-----------")

    return outputs

def cast_16(x, use_fp16=True):
    if not use_fp16:
        return x
    if isinstance(x, list) or isinstance(x, tuple):
        return [tf.cast(a, tf.float16) for a in x]
    return tf.cast(x, tf.float16)

def cast_16_wrapper(func):
    def new_func(*args, **kwargs):
        x=func(*args, **kwargs)
        return cast_16(x)
    return new_func

custom_dtype = tf.float16 if args.fp16 else tf.float32

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * tf.pow(x,3))))

def dropout(inputs, dropout_rate, training):
    return tf.layers.dropout(inputs, rate=dropout_rate, training=training)

def rms_norm(x, eps=1e-8, p=-1., bias=False, scope=None):
    with tf.variable_scope(scope or "rms_norm"):
        x = tf.cast(x, dtype=tf.float32)
        layer_size = x.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        if bias:
            offset = tf.get_variable("offset", [layer_size], initializer=tf.zeros_initializer())
        else:
            offset = 0.

        scale = tf.cast(scale, dtype=tf.float32)
        offset = tf.cast(offset, dtype=tf.float32)

        if p < 0. or p > 1.:
            ms = tf.reduce_mean(x ** 2, -1, keep_dims=True)
        else:
            partial_size = int(layer_size * p)
            partial_x, _ = tf.split(x, [partial_size, layer_size - partial_size], axis=-1)

            ms = tf.reduce_mean(partial_x ** 2, -1, keep_dims=True)

        return tf.cast(scale * x * tf.rsqrt(ms + eps) + offset, dtype=tf.float16 if args.fp16 else tf.float32)

def ffn(hidden_states,
        d_model,
        d_ff,
        dropout_rate=0.1,
        training=True,
        reuse=None):
    with tf.variable_scope("FFN", reuse=reuse):
        wi = tf.get_variable('wi', (d_model, d_ff))
        wo = tf.get_variable('wo', (d_ff, d_model))

        # pre norm
        normed_hidden_states = rms_norm(hidden_states)

        dense_output = tf.matmul(normed_hidden_states, wi)
        dense_output = gelu(dense_output)
        dense_output = dropout(dense_output, dropout_rate, training)

        dense_output = tf.matmul(dense_output, wo)
        dense_output = dropout(dense_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + dense_output

        return hidden_states

def self_attention(hidden_states,
                   d_model,
                   d_kv,
                   n_heads,
                   query_len,
                   kv_seq_len,
                   item_idx=None,
                   code_idx=None,
                   dropout_rate=0.1,
                   key_masks=None,
                   causality=False,
                   training=True,
                   reuse=None,
                   enable_kv_cache=False,
                   index=None,
                   block=None):
    with tf.variable_scope("self_attention", reuse=reuse):
        inner_dim = d_kv * n_heads

        q = tf.get_variable('q', (d_model, inner_dim))
        k = tf.get_variable('k', (d_model, inner_dim))
        v = tf.get_variable('v', (d_model, inner_dim))
        o = tf.get_variable('o', (inner_dim, d_model))

         # pre_norm
        normed_hidden_states = rms_norm(hidden_states)
        query_states = tf.matmul(normed_hidden_states, q)

        # NOTE(caikuo): KV Cache, 每个 Item 维护一个 cache，便于进行组合式检索
        # cache 结构：[(bos, sid1, sid2, sid3), ..., (bos, sid1, sid2, sid3),...]
        enable_self_att_kv_cache = args.self_att_kv_cache and enable_kv_cache
        if enable_self_att_kv_cache:
            key_states = tf.matmul(normed_hidden_states, k)
            value_states = tf.matmul(normed_hidden_states, v)
            if index > 0:
                cache_K_list = []
                cache_V_list = []
                # 1、读取前序 item 的 kv cache，更新到 cache_list
                for i in range(item_idx-1):
                    cache_K, cache_V = self_att_kv_cache[i]["self_att_kv_cache_" + str(block)]
                    cache_K_list.append(cache_K)
                    cache_V_list.append(cache_V)

                # 2、query_len 等于 2 时，token 为上一个 item 的last token + 当前 item 的 bos，需要把 last token 更新到上一个 kv cache
                if query_len == 2:
                    # 此时 item_idx 一定大于 零，不需要额外做判断
                    # 此时 code_idx 一定等于 零
                    print("check query_len==2, code_idx = ", code_idx)
                    print("check query_len==2, item_idx = ", item_idx)
                    last_K_cache, last_V_cache = self_att_kv_cache[item_idx-1]["self_att_kv_cache_" + str(block)]
                    last_K_cache = tf.concat([last_K_cache, key_states[:, 0:1, :]])
                    last_V_cache = tf.concat([last_V_cache, value_states[:, 0:1, :]])
                    self_att_kv_cache[item_idx-1]["self_att_kv_cache_" + str(block)] = [last_K_cache, last_V_cache]
                    # 创建并更新 cur item cache
                    self_att_kv_cache[item_idx]["self_att_kv_cache_" + str(block)] = [key_states[:, 1:2, :], value_states[:, 1:2, :]]

                # 3、读取当前 item 的 kv cache
                # 此段逻辑与 query_len == 2 的逻辑是互斥的
                if code_idx > 0:
                    cur_K_cache, cur_V_cache =  self_att_kv_cache[item_idx]["self_att_kv_cache_" + str(block)]
                    cache_K_list.append(cur_K_cache)
                    cache_V_list.append(cur_V_cache)
                    # 更新 cur item cache
                    self_att_kv_cache[item_idx]["self_att_kv_cache_" + str(block)] = [
                        tf.concat([cur_K_cache, key_states], axis=1),
                        tf.concat([cur_V_cache, value_states], axis=1)
                    ]

                # 4、拼接所有 cache 并更新 states
                if len(cache_K_list) > 0:
                    key_states = tf.concat(cache_K_list + [key_states], axis=1) # (N, T_k, C)
                    value_states = tf.concat([cache_V_list + [value_states], axis=1) # (N, T_k, C)
        else:
            key_states = tf.matmul(normed_hidden_states, k)
            value_states = tf.matmul(normed_hidden_states, v)

        scores = tf.matmul(tf.transpose(tf.reshape(query_states / (d_kv ** 0.5), [-1, query_len, n_heads, d_kv]), [0, 2, 1, 3]),
                           tf.transpose(tf.reshape(key_states, [-1, kv_seq_len, n_heads, d_kv]), [0, 2, 1, 3]),
                           transpose_b=True) # (bz, n_heads, query_len, kv_seq_len)

        scores = tf.reshape(scores, (-1, query_len, kv_seq_len)) # (bz * n_heads, query_len, kv_seq_len)

        # key_mask
        if key_masks is not None:
            scores = mask(scores, key_masks, type="key")

        if causality:
            scores = mask(scores, type="future")

        weights = tf.nn.softmax(scores)
        weights = dropout(weights, dropout_rate, training)

        weights = tf.reshape(weights, (-1, n_heads, query_len, kv_seq_len)) # (bz, n_heads, query_len, kv_seq_len)

        value_states = tf.transpose(tf.reshape(value_states, [-1, kv_seq_len, n_heads, d_kv]), [0, 2, 1, 3])

        attn_output = tf.matmul(weights, value_states) # (bz, n_heads, query_len, d_kv)
        attn_output = tf.squeeze(tf.concat(tf.split(attn_output, n_heads, axis=1), axis=3), axis=1)
        attn_output = tf.matmul(attn_output, o)
        attn_output = dropout(attn_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + attn_output

        return hidden_states

def cross_attention(hidden_states,
                    key_value_states,
                    d_model,
                    d_kv,
                    n_heads,
                    enc_seq_len,
                    query_len,
                    dropout_rate=0.1,
                    key_masks=None,
                    causality=False,
                    training=True,
                    reuse=None,
                    index=None,
                    block=None,):
    with tf.variable_scope("cross_attention", reuse=reuse):
        inner_dim = d_kv * n_heads

        q = tf.get_variable('q', (d_model, inner_dim))
        k = tf.get_variable('k', (d_model, inner_dim))
        v = tf.get_variable('v', (d_model, inner_dim))
        o = tf.get_variable('o', (inner_dim, d_model))

         # pre_norm
        normed_hidden_states = rms_norm(hidden_states)

        query_states = tf.matmul(normed_hidden_states, q)

        if args.cross_att_kv_cache:
            if index == 0:
                key_states = tf.matmul(key_value_states, k)
                value_states = tf.matmul(key_value_states, v)
                cross_att_kv_cache["cross_att_kv_cache_" + str(block)] = [key_states, value_states]
            else:
                key_states, value_states = cross_att_kv_cache["cross_att_kv_cache_"+str(block)]
        else:
            key_states = tf.matmul(key_value_states, k)
            value_states = tf.matmul(key_value_states, v)

        scores = tf.matmul(tf.transpose(tf.reshape(query_states / (d_kv ** 0.5), [-1, query_len, n_heads, d_kv]), [0, 2, 1, 3]),
                           tf.transpose(tf.reshape(key_states, [-1, enc_seq_len, n_heads, d_kv]), [0, 2, 1, 3]),
                           transpose_b=True) # (bz, n_heads, query_len, enc_seq_len)

        scores = tf.reshape(scores, (-1, query_len, enc_seq_len)) # (bz * n_heads, query_len, enc_seq_len)

        # key_mask
        if key_masks is not None:
            scores = mask(scores, key_masks, type="key")

        if causality:
            scores = mask(scores, type="future")

        weights = tf.nn.softmax(scores)
        weights = dropout(weights, dropout_rate, training)

        weights = tf.reshape(weights, (-1, n_heads, query_len, enc_seq_len)) # (bz, n_heads, query_len, enc_seq_len)

        value_states = tf.transpose(tf.reshape(value_states, [-1, enc_seq_len, n_heads, d_kv]), [0, 2, 1, 3])

        attn_output = tf.matmul(weights, value_states) # (bz, n_heads, query_len, d_kv)
        attn_output = tf.squeeze(tf.concat(tf.split(attn_output, n_heads, axis=1), axis=3), axis=1)
        attn_output = tf.matmul(attn_output, o)
        attn_output = dropout(attn_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + attn_output

        return hidden_states

if args.fp16:
    config.new_embedding=cast_16_wrapper(config.new_embedding)
    tf.get_variable=cast_16_wrapper(tf.get_variable)

training = False

class ModelConfig:
    def __init__(self):
        self.hidden_units = 1024
        self.d_model = 1024
        self.d_ff = 1024 * 4
        self.d_kv = 64
        self.n_heads = 8
        self.dropout_rate = 0.1
        self.num_block = 4

        self.seq_len_photo_input = 256
        self.vocab_size = 3*8192
        self.seq_len_ids_label = 3
        self.max_target_num = 5

model_config = ModelConfig()

# define kv cache
cross_att_kv_cache = {}
self_att_kv_cache = [{} for _ in range(model_config.max_target_num)] # [{kv_cache1}, {kv_cache2}, ..., {}]

lookup_table = tf.get_variable('lookup_table',
                                shape=[model_config.vocab_size, model_config.d_model],trainable=False)

# 0. hyper param
default_beam_search_num_list = [128, 128, 128]
default_pre_cut_num_list = [128, 128, 128]
beam_search_num_list = tf.reshape(tf.cast(tf.cast(config.get_extra_param("beam_search_num_list", size=3), custom_dtype), tf.int32), [-1])
beam_search_num_list = tf.where(beam_search_num_list > 0, beam_search_num_list, default_beam_search_num_list)
pre_cut_num_list = tf.reshape(tf.cast(tf.cast(config.get_extra_param("pre_cut_num_list", size=3), custom_dtype), tf.int32), [-1])
pre_cut_num_list = tf.where(pre_cut_num_list > 0, pre_cut_num_list, default_pre_cut_num_list)

# 1. seq input
inputs_ids_embeddings = config.new_embedding("inputs_ids_embeddings", dim=model_config.d_model, slots="1", expand=model_config.seq_len_photo_input, common=not training)
inputs_ids_embeddings = tf.reshape(inputs_ids_embeddings, [-1, model_config.seq_len_photo_input, model_config.d_model])

# 2. bos token
batch_size = 1
bos_token_embeddings = tf.get_variable('bos_token_embeddings', shape=[model_config.max_target_num, model_config.hidden_units], trainable=False) # (1, hidden_units)
bos_token_embeddings = tf.expand_dims(bos_token_embeddings, axis=0)
bos_token_embeddings = tf.tile(bos_token_embeddings, [batch_size, 1, 1])
bos_list = tf.split(bos_token_embeddings, model_config.max_target_num, axis=1) # [(bs, 1, dim),...]

# 3. pos emb
position_table = tf.get_variable('position_table',
                                 shape=[model_config.seq_len_photo_input, model_config.hidden_units], 
                                 trainable=False)

# 4. encode compute
enc = inputs_ids_embeddings + position_table
src_masks = None
with tf.variable_scope("encoder"):
    for i in range(model_config.num_block):
        with tf.variable_scope("num_blocks_%d" % i):
            enc = self_attention(hidden_states=enc,
                                 d_model=model_config.d_model,
                                 d_kv=model_config.d_kv,
                                 n_heads=model_config.n_heads,
                                 query_len=model_config.seq_len_photo_input,
                                 kv_seq_len=model_config.seq_len_photo_input,
                                 dropout_rate=model_config.dropout_rate,
                                 key_masks=src_masks,
                                 causality=False,
                                 training=training)

            enc = ffn(hidden_states=enc,
                      d_model=model_config.d_model,
                      d_ff=model_config.d_ff,
                      dropout_rate=model_config.dropout_rate,
                      training=training)

encoder_outputs = enc

# 5. decode compute
lookup_table_reshaped = tf.reshape(lookup_table, [3, 8192, -1])
lookup_table_list = [lookup_table_reshaped[0,:,:], lookup_table_reshaped[1,:,:], lookup_table_reshaped[2,:,:]]

infer_item_num = model_config.max_target_num
code_len = model_config.seq_len_ids_label

# 全局变量
final_decoder_result = [] # [(n, k), ..., (n, k)]
final_prob_result = [] # [(n, k), ..., (n, k)]
ori_decoder_inputs = bos_list[0] # 预估第一个 item

print_debug = []
for item_idx in range(infer_item_num):
    # 一个 item infer
    # 5.1 var which save search result
    # 局部变量，下一个 item infer 重新初始化
    decoder_result = tf.ones([1,1], dtype=tf.int32)
    prob_result = tf.ones([1,1], dtype=tf.float16 if args.fp16 else tf.float32)
    beam_search_logits = tf.ones([1], dtype=tf.float16 if args.fp16 else tf.float32)
    all_token_num = 1

    for index in range(code_len):
        dec = ori_decoder_inputs # 第一次是 [1, 1, dim]
        acc_index = item_idx * code_len + index
        kv_seq_len = acc_index + item_idx + 1
        query_len = 1
        if index == 0 and item_idx > 0:
            query_len = 2
        print_debug.append(kv_seq_len)
# 5.2 att 计算
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            for i in range(model_config.num_block):
                with tf.variable_scope("num_blocks_%d" % i):
                    dec = self_attention(hidden_states=dec,
                                        d_model=model_config.d_model,
                                        d_kv=model_config.d_kv,
                                        n_heads=model_config.n_heads,
                                        query_len=query_len,
                                        item_idx=item_idx,
                                        code_idx=index,
                                        kv_seq_len=kv_seq_len,
                                        dropout_rate=model_config.dropout_rate,
                                        key_masks=None,
                                        causality=False,
                                        training=training,
                                        reuse=tf.AUTO_REUSE,
                                        enable_kv_cache=True,
                                        index=acc_index,
                                        block=i)

                    # cross attention KV 优化
                    dec = cross_attention(hidden_states=dec,
                                        key_value_states=encoder_outputs,
                                        d_model=model_config.d_model,
                                        d_kv=model_config.d_kv,
                                        n_heads=model_config.n_heads,
                                        enc_seq_len=model_config.seq_len_photo_input,
                                        query_len=query_len,
                                        dropout_rate=model_config.dropout_rate,
                                        key_masks=src_masks,
                                        causality=False,
                                        training=training,
                                        reuse=tf.AUTO_REUSE,
                                        index=acc_index,
                                        block=i)

                    dec = ffn(hidden_states=dec,
                            d_model=model_config.d_model,
                            d_ff=model_config.d_ff,
                            dropout_rate=model_config.dropout_rate,
                            training=training,
                            reuse=tf.AUTO_REUSE)

# 5.3 topk 采样
        # last token out
        decoder_outputs = dec
        decoder_outputs_last = decoder_outputs[:,-1,:] # [n, dim] 第一次n=1, 之后n=beam_search_num
        decoder_outputs_last = decoder_outputs_last*tf.cast(tf.sqrt(1.0 / model_config.d_model),tf.float16 if args.fp16 else tf.float32)

        token_table = lookup_table_list[index] # [4096,dim]
        logits = tf.matmul(decoder_outputs_last, token_table, transpose_b=True)
        probs = tf.nn.softmax(logits) #[n, 4096] 把概率保存下来输出

# 5.4 precut
        # NOTE(caikuo): 前3层完整 beam search，后3层 precut 简化计算
        pre_cut_num = pre_cut_num_list[index]
        beam_search_num = beam_search_num_list[index]
        topk_probs, topk_indices = tf.nn.top_k(probs, k=pre_cut_num) # [n, k]
        topk_values = tf.math.log(topk_probs)

        # NOTE(caikuo): 这里不要 gather 出来，参数量太大了，直接从 lookup_table gather
        topk_embeddings = tf.nn.embedding_lookup(token_table, topk_indices) # [n, k, dim]

# 5.5 compute topk
        topk_values = tf.reshape(topk_values,[-1]) # [n*k]
        beam_search_logits = tf.expand_dims(beam_search_logits, axis=1) # [n,1]
        beam_search_logits = tf.tile(beam_search_logits, [1, pre_cut_num]) # [n, k]
        beam_search_logits = tf.reshape(beam_search_logits, [-1]) # [n*k]
        beam_search_logits = beam_search_logits + topk_values
        beam_search_logits, beam_search_top_k_indices = tf.nn.top_k(beam_search_logits, k=beam_search_num) # [n]

# 5.6 update kv cache
        if args.self_att_kv_cache:
            kv_indices = tf.cast(tf.floor(beam_search_top_k_indices / pre_cut_num), tf.int32) # [n]
            for i in range(model_config.num_block):
                cache_K, cache_V = self_att_kv_cache[item_idx]["self_att_kv_cache_" + str(i)]
                cache_K = tf.gather(cache_K, kv_indices)
                cache_V = tf.gather(cache_V, kv_indices)
                self_att_kv_cache[item_idx]["self_att_kv_cache_" + str(i)] = [cache_K, cache_V]
                #print("self_att_kv_cache: ", self_att_kv_cache["self_att_kv_cache_" + str(i)])

# 5.6 update output
        decoder_result = tf.expand_dims(decoder_result, axis=1) # [n, 1, all_token_num]
        decoder_result = tf.tile(decoder_result, [1, pre_cut_num, 1]) # [n, k, all_token_num]
        topk_indices = tf.expand_dims(topk_indices, axis=-1) # [n, k, 1]
        decoder_result_concat = tf.concat([decoder_result, topk_indices], axis=-1) #[n, k, all_token_num+1]
        decoder_result = tf.reshape(decoder_result_concat, [-1, all_token_num+1]) #[n*k, all_token_num+1]
        decoder_result = tf.gather(decoder_result, beam_search_top_k_indices) # [k, all_token_num+1]

        # concat prob
        prob_result = tf.expand_dims(prob_result, axis=1) # [n, 1, all_token_num]
        prob_result = tf.tile(prob_result, [1, pre_cut_num, 1]) # [n, k, all_token_num]
        topk_probs = tf.expand_dims(topk_probs, axis=-1)
        prob_result_concat = tf.concat([prob_result, topk_probs], axis=-1) # [n, k, all_token_num+1]
        prob_result = tf.reshape(prob_result_concat, [-1, all_token_num+1]) #[n*k, all_token_num+1]
        prob_result = tf.gather(prob_result, beam_search_top_k_indices) # [k, all_token_num+1]

# 5.7 构造 next step input
        topk_embeddings = tf.reshape(topk_embeddings,[-1,model_config.d_model]) # [n*k, dim]
        topk_embeddings = tf.expand_dims(topk_embeddings,axis=1) # [n*k, 1, dim]
        ori_decoder_inputs = tf.gather(topk_embeddings, beam_search_top_k_indices)
        if (index == code_len-1 and (item_idx != infer_item_num-1)): # last token per item
            bos_t = tf.tile(bos_list[item_idx+1], [beam_search_num, 1, 1])
            ori_decoder_inputs = tf.concat([ori_decoder_inputs, bos_t], axis=1)

        print("ori_decoder_inputs: ", ori_decoder_inputs)
        all_token_num = all_token_num + 1

    # 一个 item infer 完成
    final_decoder_result.append(decoder_result)
    final_prob_result.append(prob_result)

decoder_result = decoder_result[:, 1:] # [k, infer_item_num * 6]
decoder_result_output = tf.reshape(decoder_result,[1, -1]) # [1, k * infer_item_num * 6]
prob_result = prob_result[:, 1:] # [k, 6]
prob_result_output = tf.reshape(prob_result,[1, -1]) # [k,6]
beam_search_logits_output = tf.reshape(beam_search_logits,[1, -1])
print(print_debug)

targets = [
    ("decoder_result_output", tf.cast(decoder_result_output, dtype=tf.float32)),
    ("prob_result_output", tf.cast(prob_result_output, dtype=tf.float32)),
    ("beam_search_logits_output", tf.cast(beam_search_logits_output, dtype=tf.float32)),
]

config.dump_predict_config(
    "./" + args.dest_dir,
    targets,
    input_type=3,
    dump_mode=args.mode,
    text=True
)
