from __future__ import print_function
import numpy as np

import os
import sys
import logging
import argparse
import functools
import math
import contextlib

from modules import self_attention, cross_attention, ffn, rms_norm
from utils.feature_importance import FeatureImportanceUtil
from moe import MoeLayer, ModelArgs

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'user_predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--with_kai_v2', default=True)
args = parser.parse_known_args()[0]

# srpo_loss_weight = 10.0
class ModelConfig:
    def __init__(self):
        self.d_model = 1024
        self.d_ff = 1024 * 4
        self.d_kv = 64
        self.n_heads = 8
        self.dropout_rate = 0.1
        self.num_block = 4

        self.hidden_units = 1024
        self.seq_len_ids_label = 3 # semantic id 数量
        self.seq_len_photo_input = 256 # 序列长度

        self.vocab_size = 3*8192
        self.vocab_dim = 8192

        self.max_target_num = 5 # ListGEN 模式下，max target num 数量拉满

model_config = ModelConfig()

model_args: ModelArgs = ModelArgs(
    dim=1024,
    #n_layers=2,
    #n_heads=1,
    #n_kv_heads: Optional[int] = None,

    multiple_of=128,  # make SwiGLU hidden layer size multiple of large power of 2
    #ffn_dim_multiplier: Optional[float] = None
    #norm_eps: float = 1e-5

    num_experts=24,
    moe_top_k=2,
    is_perf_moe_metric=True
)

if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                   access_method=config.nn.ProbabilityAccess(100.0),
                                   recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=120, delete_threshold=2.0, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    # 注册样本过滤逻辑 ref: https://docs.corp.kuaishou.com/d/home/fcAAgCXWs240U2r3PFFhnB3Fv
    def filter_mask_wrapper(dataset):
        # 1. 声明字段
        #  sample_type为字段名，特征类型dataset.DENSE表示稠密，tf.int64为数据类型，dim为1
        # dataset.add_feature('is_high_value_session_v2', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('session_item_num', dataset.DENSE, tf.int64, 1)
        dataset.add_feature("is_downgrade", dataset.DENSE, tf.int64, 1)
        # 2.声明mask，batch是一个dict，key为声明的字段名，value根据特征类型分为2种情况：
        # dataset.DENSE: 值为tf.Tensor
        # dataset.SPARSE: 值为元组: (tf.Tensor, tf.Tensor)，
        #   其中第一个tensor表示feasign，第二个tensor表示cumsum
        #   可以使用tf.RaggedTensor.from_row_splits转成RaggedTensor
        def mask_fn(batch):
            # is_high_value_session_v2 = batch['is_high_value_session_v2']
            session_item_num = batch['session_item_num']
            #mask = tf.math.equal(is_high_value_session, 0)
            # mask = tf.math.logical_or(tf.math.equal(is_high_value_session_v2, 0),
            #                           tf.math.less(session_item_num, model_config.max_target_num))
            
            mask = tf.math.less(session_item_num, model_config.max_target_num)
            is_downgrade = batch["is_downgrade"]
            mask_downgrade = tf.math.equal(is_downgrade, 1)
            return tf.math.logical_or(mask, mask_downgrade)
        # 3.返回mask_fn
        return mask_fn

    # 注册过滤条件
    config.declare_sample_filter(filter_mask_wrapper, data_source_name='train')
    #config.declare_sample_filter(filter_mask_wrapper, data_source_name='test')
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    if not args.dryrun and not args.with_kai:
        # monkey patch
        import mio_tensorflow.patch as mio_tensorflow_patch

        mio_tensorflow_patch.apply()

    import tensorflow as tf
    from mio_tensorflow.config import MioConfig

    sys.path.append("../../../../")

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

def conditional_tf_print(condition_func, *args, **kwargs):
    return tf.cond(
        condition_func(),
        lambda: tf.print(*args, **kwargs),
        lambda: tf.no_op(),
    )

def cast32(t):
    return tf.cast(t, tf.float32)

use_xla = False
def new_xla_jit_context():
    if use_xla:
        return tf.xla.experimental.jit_scope()
    else:
        return contextlib.suppress()

feature_importance_util = FeatureImportanceUtil(config)

training = args.mode == "train"
print_ops = []

# target 侧：先扩 batch
lookup_table = tf.get_variable('lookup_table',
                                dtype=tf.float32,
                                shape=[model_config.vocab_size, model_config.hidden_units],)

label_ids = config.get_dense_fea("session_label_v2", dim=model_config.seq_len_ids_label*model_config.max_target_num, dtype=tf.int64) # (bz, 3 * max_target_num)
session_item_num = tf.cast(config.get_dense_fea("session_item_num", dim=1, dtype=tf.int64), dtype=tf.float32)
is_high_value_session = config.get_dense_fea("is_high_value_session", dim=1, dtype=tf.int64)
session_item_num = tf.minimum(session_item_num, model_config.max_target_num) # (bz, 1)
batch_size = tf.shape(session_item_num)[0]

# 调参
start_idx = 32 * 1
universal_weight = config.get_dense_fea("universal_param", dim=start_idx+32, dtype=tf.float32)[:,start_idx:start_idx+32]
sft_w = tf.reduce_mean(universal_weight[:,0])
srpo_w = tf.reduce_mean(universal_weight[:,1])
dpo_w = tf.reduce_mean(universal_weight[:,2])
vtr_w = tf.reduce_mean(universal_weight[:,3])
lsst_w = tf.reduce_mean(universal_weight[:,4])
swpst_w = tf.reduce_mean(universal_weight[:,5])
evtr_w = tf.reduce_mean(universal_weight[:,6])
lvtr_w = tf.reduce_mean(universal_weight[:,7])
ltr_w = tf.reduce_mean(universal_weight[:,8])
wtr_w = tf.reduce_mean(universal_weight[:,9])
cmtr_w = tf.reduce_mean(universal_weight[:,10])
setr_w = tf.reduce_mean(universal_weight[:,11])
ftr_w = tf.reduce_mean(universal_weight[:,12])
cltr_w = tf.reduce_mean(universal_weight[:,13])
prof_w = tf.reduce_mean(universal_weight[:,14])
left_w = tf.reduce_mean(universal_weight[:,15])
cmef_w = tf.reduce_mean(universal_weight[:,16])
epst_w = tf.reduce_mean(universal_weight[:,17])
vtr2_w = tf.reduce_mean(universal_weight[:,18])
play_w = tf.reduce_mean(universal_weight[:,19])

ids_bias = [0, 8192, 8192*2] * model_config.max_target_num
ids_bias = tf.convert_to_tensor(ids_bias, tf.int64)
ids_bias = tf.reshape(ids_bias, [1, -1]) # (1, 3 * max_target_num)

label_ids_shifted = tf.cast(label_ids, tf.int64) + ids_bias # (bz, 3 * max_target_num)
label_ids_shifted = tf.cast(label_ids_shifted, tf.int32)
label_ids_embeddings = tf.nn.embedding_lookup(lookup_table, label_ids_shifted) # (bz, 3 * max_target_num, hidden_units)

bos_token_embeddings = tf.get_variable('bos_token_embeddings', dtype=tf.float32, shape=[model_config.max_target_num, model_config.hidden_units]) # (1, hidden_units)
bos_token_embeddings = tf.tile(tf.expand_dims(bos_token_embeddings, axis=0), [batch_size, 1, 1]) # (bz, 1, hidden_units)

# user 侧
input_item_embeddings = feature_importance_util.config_new_embedding("input_item_embeddings", dim=model_config.hidden_units, slots="1", expand=model_config.seq_len_photo_input, common=not training)
input_item_embeddings = tf.reshape(input_item_embeddings, [-1, model_config.seq_len_photo_input, model_config.hidden_units]) # （bz, seq_len, hidden_units)
tf.summary.histogram('sparse', input_item_embeddings[:, 0, :])

# 绝对位置编码
position_table = tf.get_variable('position_table',
                                dtype=tf.float32,
                                shape=[model_config.seq_len_photo_input, model_config.hidden_units],)
src_masks = None

enc = input_item_embeddings + position_table
with tf.variable_scope("encoder"), new_xla_jit_context():
    for i in range(model_config.num_block):
        with tf.variable_scope("num_blocks_%d" % i):
            enc = self_attention(hidden_states=enc,
                                 d_model=model_config.d_model,
                                 d_kv=model_config.d_kv,
                                 n_heads=model_config.n_heads,
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

# decoder
bos_list = tf.split(bos_token_embeddings, model_config.max_target_num, axis=1) # (bs, 1, dim)
label_ids_embeddings_list = tf.split(label_ids_embeddings, model_config.max_target_num, axis=1) # (bs, 3, dim)
decoder_inputs = tf.concat([bos_list[0], label_ids_embeddings_list[0]], axis=1)
for i in range(1, model_config.max_target_num):
    decoder_inputs = tf.concat([decoder_inputs, bos_list[i], label_ids_embeddings_list[i]], axis=1)
dec = decoder_inputs

with tf.variable_scope("decoder"), new_xla_jit_context():
    for i in range(model_config.num_block):
        with tf.variable_scope("num_blocks_%d" % i):
            dec = self_attention(hidden_states=dec,
                                 d_model=model_config.d_model,
                                 d_kv=model_config.d_kv,
                                 n_heads=model_config.n_heads,
                                 dropout_rate=model_config.dropout_rate,
                                 key_masks=None,
                                 causality=True,
                                 training=training)

            if i== model_config.num_block -1:
                dec = cross_attention(hidden_states=dec,
                                    key_value_states=encoder_outputs,
                                    d_model=model_config.d_model,
                                    d_kv=model_config.d_kv,
                                    n_heads=model_config.n_heads,
                                    dropout_rate=0,
                                    key_masks=src_masks,
                                    causality=False,
                                    training=training)
            else:
                dec = cross_attention(hidden_states=dec,
                                    key_value_states=encoder_outputs,
                                    d_model=model_config.d_model,
                                    d_kv=model_config.d_kv,
                                    n_heads=model_config.n_heads,
                                    dropout_rate=model_config.dropout_rate,
                                    key_masks=src_masks,
                                    causality=False,
                                    training=training)

            #dec = ffn(hidden_states=dec,
            #          d_model=model_config.d_model,
            #          d_ff=model_config.d_ff,
            #          dropout_rate=model_config.dropout_rate,
            #          training=training)
            dec = dec + MoeLayer(model_args, name="moe_ffn_%d" % i) (rms_norm(dec))
    dec = rms_norm(dec)

decoder_outputs = dec # (bz, 6 * max_target_num, hidden_units)
decoder_inputs = dec

if training:
    with new_xla_jit_context():
        lookup_table_reshaped = tf.reshape(lookup_table, [3, 8192, -1])
        lookup_table_1 = lookup_table_reshaped[0, :, :]
        lookup_table_2 = lookup_table_reshaped[1, :, :]
        lookup_table_3 = lookup_table_reshaped[2, :, :]

        # scale
        #decoder_inputs = decoder_inputs * tf.sqrt(1.0 / model_config.d_model)

        # ListGEN 模式下 decoder 输出排列 [s1,s2,s3,s4,s5,s6, ......,s1,s2,s3,s4,s5,s6]
        # group
        decoder_outputs_1 = decoder_inputs[:, 0::4, :] # (bz, max_target_num, d_model), [s1,s1,..., s1]
        decoder_outputs_2 = decoder_inputs[:, 1::4, :] # [s2,s2,..., s2]
        decoder_outputs_3 = decoder_inputs[:, 2::4, :]

        logits_1 = tf.matmul(decoder_outputs_1, lookup_table_1, transpose_b=True) # (bz, max_target_num, 4096)
        logits_2 = tf.matmul(decoder_outputs_2, lookup_table_2, transpose_b=True)
        logits_3 = tf.matmul(decoder_outputs_3, lookup_table_3, transpose_b=True)

        # 各层的 ground truth
        label_ids = tf.cast(label_ids, tf.int32)
        gt_1 = label_ids[:, 0::3] # (bz, max_target_num)
        gt_2 = label_ids[:, 1::3]
        gt_3 = label_ids[:, 2::3]

        # 计算各层 loss
        loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_1, logits=logits_1) # (bz, max_target_num)
        loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_2, logits=logits_2) # (bz, max_target_num)
        loss_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_3, logits=logits_3) # (bz, max_target_num)

        # session relative policy optimization  
        #reward 计算
        item_all_evtr = tf.cast(config.get_dense_fea("prm_all_evtr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_all_lvtr = tf.cast(config.get_dense_fea("prm_all_lvtr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_ltr = tf.cast(config.get_dense_fea("prm_ltr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_wtr = tf.cast(config.get_dense_fea("prm_wtr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_vtr = config.get_dense_fea("prm_vtr", dim=model_config.max_target_num, dtype=tf.float32)
        item_lsst = tf.cast(config.get_dense_fea("prm_lsst", dim=model_config.max_target_num, dtype=tf.int64), dtype=tf.float32)
        item_swpst = tf.cast(config.get_dense_fea("prm_swpst", dim=model_config.max_target_num, dtype=tf.int64), dtype=tf.float32)
        item_cmtr = tf.cast(config.get_dense_fea("prm_cmtr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_setr = tf.cast(config.get_dense_fea("prm_setr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_ftr = tf.cast(config.get_dense_fea("prm_ftr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_cltr = tf.cast(config.get_dense_fea("prm_cltr", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_profile = tf.cast(config.get_dense_fea("prm_profile", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_left_slide = tf.cast(config.get_dense_fea("prm_left_slide", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_cmef = tf.cast(config.get_dense_fea("prm_cmef", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_epst = tf.cast(config.get_dense_fea("prm_epst", dim=model_config.max_target_num, dtype=tf.int64), tf.float32)
        item_vtr_v2 = config.get_dense_fea("prm_vtr_v2", dim=model_config.max_target_num, dtype=tf.float32)
        item_play_ratio = config.get_dense_fea("prm_play_ratio", dim=model_config.max_target_num, dtype=tf.float32)
        item_vtr = tf.clip_by_value(item_vtr / 40, 0, 1)
        item_lsst = tf.clip_by_value(item_lsst / 30, 0, 1)
        item_swpst = tf.clip_by_value(item_swpst / 20, 0, 1)
        item_vtr_v2 = tf.clip_by_value(item_vtr_v2 / 40, 0, 1)

        # item_duration = tf.clip_by_value((item_vtr + item_lsst + item_swpst) / 90, 0, 1)
        item_reward = item_vtr * vtr_w \
                    + item_lsst * lsst_w \
                    + item_swpst * swpst_w \
                    + item_all_evtr * evtr_w \
                    + item_all_lvtr * lvtr_w \
                    + item_ltr * ltr_w \
                    + item_wtr * wtr_w \
                    + item_cmtr * cmtr_w \
                    + item_setr * setr_w \
                    + item_ftr * ftr_w \
                    + item_cltr * cltr_w \
                    + item_profile * prof_w \
                    + item_left_slide * left_w \
                    + item_cmef * cmef_w \
                    + item_epst * epst_w \
                    + item_vtr_v2 * vtr2_w \
                    + item_play_ratio * play_w
        reward_clip_value = 20
        item_reward = tf.clip_by_value(item_reward, -reward_clip_value, reward_clip_value)

        # srpo loss
        # 计算每个样本的概率
        prob_1 = tf.nn.softmax(logits_1, axis=-1) # (bz, max_target_num)
        prob_2 = tf.nn.softmax(logits_2, axis=-1) # (bz, max_target_num) 
        prob_3 = tf.nn.softmax(logits_3, axis=-1) # (bz, max_target_num)
        # 根据gt取出对应的概率
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, model_config.max_target_num]) # (bz, max_target_num)
        seq_indices = tf.tile(tf.expand_dims(tf.range(model_config.max_target_num), 0), [batch_size, 1]) # (bz, max_target_num)
        
        # 使用gather_nd取出每层对应gt的概率
        prob_1_selected = tf.gather_nd(
            prob_1,
            tf.stack([batch_indices, seq_indices, gt_1], axis=-1)
        ) # (bz, max_target_num)
        
        prob_2_selected = tf.gather_nd(
            prob_2, 
            tf.stack([batch_indices, seq_indices, gt_2], axis=-1)
        ) # (bz, max_target_num)
        
        prob_3_selected = tf.gather_nd(
            prob_3,
            tf.stack([batch_indices, seq_indices, gt_3], axis=-1)
        ) # (bz, max_target_num)
        
        # 连乘得到每个位置的总概率
        item_prob = prob_1_selected * prob_2_selected * prob_3_selected # (bz, max_target_num)
        
        # 计算每个样本的log概率
        log_prob = tf.math.log(item_prob + 1e-10) # 加一个小值避免log(0)
        
        # 计算advantage
        mean_reward = tf.reduce_mean(item_reward, axis=1, keepdims=True) # (bz, 1)
        
        # 计算方差和标准差
        n = tf.cast(tf.shape(item_reward)[1], item_reward.dtype)
        diff_sq = tf.square(item_reward - mean_reward)
        sum_diff_sq = tf.reduce_sum(diff_sq, axis=1, keepdims=True)
        var_reward = sum_diff_sq / (n - 1)
        std_reward = tf.sqrt(var_reward + 1e-8) # 加一个小值避免除0
        

        # 标准化advantage
        advantage = (item_reward - mean_reward) / std_reward # (bz, max_target_num)
        
       
        # 计算srpo loss
        log_prob_detach = tf.stop_gradient(log_prob)
        epsilon = 0.2
        ratio = tf.exp(log_prob - log_prob_detach)
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
        srpo_loss = -tf.minimum(ratio * advantage, clipped_ratio * advantage)

        # mask 操作
        # mask 的 shape 是各层共享的, shape = (bz, 9)
        # 第一版可以在 pipeline 里做限定，必须满足 max_target_num
        label_matrix = tf.tile(tf.expand_dims(tf.range(model_config.max_target_num, dtype=tf.float32), 0), [batch_size, 1]) # (bz, max_target_num)
        label_matrix = tf.where(label_matrix < session_item_num, tf.ones_like(label_matrix), tf.zeros_like(label_matrix)) # (bz, max_target_num)

        # mask loss
        # 这里除以 batch size 待确认？？？
        bs_float32 = tf.cast(batch_size, tf.float32)
        #weighted_loss_1 = tf.losses.compute_weighted_loss(loss_1, label_matrix, reduction='weighted_sum') / bs_float32
        #weighted_loss_2 = tf.losses.compute_weighted_loss(loss_2, label_matrix, reduction='weighted_sum') / bs_float32
        #weighted_loss_3 = tf.losses.compute_weighted_loss(loss_3, label_matrix, reduction='weighted_sum') / bs_float32
        #weighted_loss_4 = tf.losses.compute_weighted_loss(loss_4, label_matrix, reduction='weighted_sum') / bs_float32
        #weighted_loss_5 = tf.losses.compute_weighted_loss(loss_5, label_matrix, reduction='weighted_sum') / bs_float32
        #weighted_loss_6 = tf.losses.compute_weighted_loss(loss_6, label_matrix, reduction='weighted_sum') / bs_float32

        weighted_loss_1 = tf.reduce_sum(loss_1 * label_matrix) / bs_float32
        weighted_loss_2 = tf.reduce_sum(loss_2 * label_matrix) / bs_float32
        weighted_loss_3 = tf.reduce_sum(loss_3 * label_matrix) / bs_float32
        srpo_loss = tf.reduce_sum(srpo_loss * label_matrix) / bs_float32

        train_step = config.get_step()
        srpo_loss = srpo_loss * srpo_w

        sft_loss = weighted_loss_1 + weighted_loss_2 + weighted_loss_3
        sft_loss = sft_loss * sft_w
        total_loss = sft_loss + srpo_loss

        # 分析 grad norm
        sft_grad = tf.gradients(sft_loss, decoder_inputs)[0]
        srpo_grad = tf.gradients(srpo_loss, decoder_inputs)[0]

        sft_grad_norm = tf.norm(sft_grad)
        srpo_grad_norm = tf.norm(srpo_grad)
        tf.summary.scalar("grad_norm/sft_grad_norm", sft_grad_norm)
        tf.summary.scalar("grad_norm/srpo_grad_norm", srpo_grad_norm)

        tf.summary.scalar("srpo/srpo_loss", srpo_loss)
        tf.summary.scalar("srpo/sft_loss", sft_loss)
        tf.summary.scalar("srpo/total_loss", total_loss)
        tf.summary.scalar("srpo/advantage_max", tf.reduce_max(advantage))
        tf.summary.scalar("srpo/advantage_min", tf.reduce_min(advantage))
        tf.summary.scalar("srpo/ratio_mean", tf.reduce_mean(ratio))
        tf.summary.scalar("srpo/reward_mean", tf.reduce_mean(item_reward))
        tf.summary.scalar("srpo/reward_std", tf.reduce_mean(std_reward))


        print_debug = conditional_tf_print(
            lambda: tf.equal(tf.mod(train_step, 4000), 1),
            "step: ",train_step,
            "origin_decoder_outputs", decoder_outputs[0,:,:],
            "scale_decoder_outputs", decoder_inputs[0,:,:],
            "input_label_ids_embeddings",label_ids_embeddings[0],
            "grad_lookup_table", tf.gradients(total_loss, lookup_table_1)[0],
            "grad_lookup_table_max", tf.reduce_max(tf.gradients(total_loss, lookup_table_1)[0]),
            "look_up_table", lookup_table,
            "logits_1", logits_1[0,:,:],
            "input_item_embeddings", input_item_embeddings[0,:,:],
            "grad_input_item_embeddings", tf.gradients(total_loss, input_item_embeddings)[0][0,:,:],
            "grad_input_max", tf.reduce_max(tf.gradients(total_loss, input_item_embeddings)[0]),
            output_stream=sys.stdout,
            summarize=5,
        )
        print_ops.append(print_debug)

        # 评估流程, 写个函数复用
        def eval_acc(logits, labels, mask):
            '''
            logits: (bs, dim) float32
            labels: (bs, 1) int32
            mask: (bs, 1) float32
            '''
            m = tf.argmax(logits, axis=1) # (bs,)
            l = tf.cast(labels, tf.int64) # (bs,)
            acc = tf.reduce_sum(mask * cast32(tf.equal(m, l))) / tf.reduce_sum(mask)
            return acc

        def eval_hit_rate(logits, labels, mask):
            '''
            logits: (bs, n, dim) float32
            labels: (bs, n) int32
            mask: (bs,) float32
            '''
            h = tf.argmax(logits, axis=2)
            l = tf.cast(labels, tf.int64)
            hit = tf.equal(h, l)
            hit = tf.reduce_min(tf.cast(hit, tf.float32), axis=1) * mask
            hit = tf.reduce_sum(hit) / tf.reduce_sum(mask)
            return hit

        acc_1_1 = eval_acc(logits_1[:, 0, :], gt_1[:, 0], label_matrix[:, 0])
        acc_1_2 = eval_acc(logits_2[:, 0, :], gt_2[:, 0], label_matrix[:, 0])
        acc_1_3 = eval_acc(logits_3[:, 0, :], gt_3[:, 0], label_matrix[:, 0])
        hit_1 = eval_hit_rate(
            logits=tf.reshape(
                tf.concat([logits_1[:, 0, :], logits_2[:, 0, :], logits_3[:, 0, :]], axis=1),
                (-1, model_config.seq_len_ids_label, model_config.vocab_dim)
            ),
            labels=tf.concat([gt_1[:, 0:1], gt_2[:, 0:1], gt_3[:, 0:1]], axis=1),
            mask=label_matrix[:, 0]
        )

        acc_2_1 = eval_acc(logits_1[:, 1, :], gt_1[:, 1], label_matrix[:, 1])
        acc_2_2 = eval_acc(logits_2[:, 1, :], gt_2[:, 1], label_matrix[:, 1])
        acc_2_3 = eval_acc(logits_3[:, 1, :], gt_3[:, 1], label_matrix[:, 1])
        hit_2 = eval_hit_rate(
            logits=tf.reshape(
                tf.concat([logits_1[:, 1, :], logits_2[:, 1, :], logits_3[:, 1, :]], axis=1),
                (-1, model_config.seq_len_ids_label, model_config.vocab_dim)
            ),
            labels=tf.concat([gt_1[:, 1:2], gt_2[:, 1:2], gt_3[:, 1:2]], axis=1),
            mask=label_matrix[:, 1]
        )

        acc_3_1 = eval_acc(logits_1[:, 2, :], gt_1[:, 2], label_matrix[:, 2])
        acc_3_2 = eval_acc(logits_2[:, 2, :], gt_2[:, 2], label_matrix[:, 2])
        acc_3_3 = eval_acc(logits_3[:, 2, :], gt_3[:, 2], label_matrix[:, 2])
        hit_3 = eval_hit_rate(
            logits=tf.reshape(
                tf.concat([logits_1[:, 2, :], logits_2[:, 2, :], logits_3[:, 2, :]], axis=1),
                (-1, model_config.seq_len_ids_label, model_config.vocab_dim)
            ),
            labels=tf.concat([gt_1[:, 2:3], gt_2[:, 2:3], gt_3[:, 2:3]], axis=1),
            mask=label_matrix[:, 2]
        )

        acc_4_1 = eval_acc(logits_1[:, 3, :], gt_1[:, 3], label_matrix[:, 3])
        acc_4_2 = eval_acc(logits_2[:, 3, :], gt_2[:, 3], label_matrix[:, 3])
        acc_4_3 = eval_acc(logits_3[:, 3, :], gt_3[:, 3], label_matrix[:, 3])
        hit_4 = eval_hit_rate(
            logits=tf.reshape(
                tf.concat([logits_1[:, 3, :], logits_2[:, 3, :], logits_3[:, 3, :]], axis=1),
                (-1, model_config.seq_len_ids_label, model_config.vocab_dim)
            ),
            labels=tf.concat([gt_1[:, 3:4], gt_2[:, 3:4], gt_3[:, 3:4]], axis=1),
            mask=label_matrix[:, 3]
        )

        acc_5_1 = eval_acc(logits_1[:, 4, :], gt_1[:, 4], label_matrix[:, 4])
        acc_5_2 = eval_acc(logits_2[:, 4, :], gt_2[:, 4], label_matrix[:, 4])
        acc_5_3 = eval_acc(logits_3[:, 4, :], gt_3[:, 4], label_matrix[:, 4])
        hit_5 = eval_hit_rate(
            logits=tf.reshape(
                tf.concat([logits_1[:, 4, :], logits_2[:, 4, :], logits_3[:, 4, :]], axis=1),
                (-1, model_config.seq_len_ids_label, model_config.vocab_dim)
            ),
            labels=tf.concat([gt_1[:, 4:5], gt_2[:, 4:5], gt_3[:, 4:5]], axis=1),
            mask=label_matrix[:, 4]
        )

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.0002)
        dense_optimizer = config.optimizer.Adam(0.0002)
        sparse_optimizer.set_replace_nan_grad(True)
        dense_optimizer.set_replace_nan_grad(True)
        sparse_optimizer.minimize(total_loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
        dense_optimizer.minimize(total_loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
    else:
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        opt = optimizer.minimize(total_loss)

    with tf.control_dependencies(print_ops):
        total_loss = tf.identity(total_loss)

    zeros = tf.fill([batch_size, 1], 0.0)
    ones = tf.fill([batch_size, 1], 1.0)

    eval_targets = []
    eval_targets += [
        ("total_loss", total_loss * ones, zeros, ones, "linear_regression"),
        ("srpo_loss", srpo_loss * ones, zeros, ones, "linear_regression"),
        ("acc_1_1", acc_1_1 * ones, zeros, ones, "linear_regression"),
        ("acc_2_1", acc_2_1 * ones, zeros, ones, "linear_regression"),
        ("acc_3_1", acc_3_1 * ones, zeros, ones, "linear_regression"),
        ("acc_4_1", acc_4_1 * ones, zeros, ones, "linear_regression"),
        ("acc_5_1", acc_5_1 * ones, zeros, ones, "linear_regression"),

        ("acc_1_2", acc_1_2 * ones, zeros, ones, "linear_regression"),
        ("acc_2_2", acc_2_2 * ones, zeros, ones, "linear_regression"),
        ("acc_3_2", acc_3_2 * ones, zeros, ones, "linear_regression"),
        ("acc_4_2", acc_4_2 * ones, zeros, ones, "linear_regression"),
        ("acc_5_2", acc_5_2 * ones, zeros, ones, "linear_regression"),

        ("acc_1_3", acc_1_3 * ones, zeros, ones, "linear_regression"),
        ("acc_2_3", acc_2_3 * ones, zeros, ones, "linear_regression"),
        ("acc_3_3", acc_3_3 * ones, zeros, ones, "linear_regression"),
        ("acc_4_3", acc_4_3 * ones, zeros, ones, "linear_regression"),
        ("acc_5_3", acc_5_3 * ones, zeros, ones, "linear_regression"),

        ("hit_1", hit_1 * ones, zeros, ones, "linear_regression"),
        ("hit_2", hit_2 * ones, zeros, ones, "linear_regression"),
        ("hit_3", hit_3 * ones, zeros, ones, "linear_regression"),
        ("hit_4", hit_4 * ones, zeros, ones, "linear_regression"),
        ("hit_5", hit_5 * ones, zeros, ones, "linear_regression"),
    ]

    # print all variables and gradients
    variables = tf.trainable_variables()
    grads = tf.gradients(total_loss, variables)
    for grad, var in list(zip(grads, variables)):
        tf.summary.scalar("variable/" + var.name + "_mean", tf.reduce_mean(var))
        tf.summary.scalar("variable/" + var.name + "_max", tf.reduce_max(var))
        tf.summary.histogram("variable/" + var.name, var)
        if grad is not None:
            tf.summary.histogram("variable/" + var.name + "_gradient", grad)
            tf.summary.scalar("variable/" + var.name + "_gradient_mean", tf.reduce_mean(grad))
            tf.summary.scalar("variable/" + var.name + "_gradient_max", tf.reduce_max(grad))

    if args.dryrun:
        config.mock_and_profile(opt, "./training_log/", batch_sizes=[512])
    else:
        if args.with_kai:
            config.dump_kai_training_config(
                "./training/conf",
                eval_targets,
                loss=total_loss,
                text=args.text,
                extra_ops=print_ops,
            )
        elif args.with_kai_v2:
            config.build_model(optimizer=[sparse_optimizer, dense_optimizer], metrics=eval_targets)
        else:
            config.dump_training_config(
                "./training/conf",
                eval_targets,
                opts=[opt],
                text=args.text,
            )
