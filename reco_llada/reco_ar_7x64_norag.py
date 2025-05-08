from __future__ import print_function
import os
import sys
import logging
import argparse
import functools
import contextlib
import random
import numpy as np
import math
parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', choices=['train', 'predict', 'gsu_offline', 'gsu'], dest='mode', default='train')
parser.add_argument('--dryrun', action="store_true")
parser.add_argument('--text', action="store_true")
parser.add_argument('--trt_transformer_fallback', action="store_true")
parser.add_argument('--no_kai_v2', dest="with_kai_v2", action="store_false")
parser.add_argument('--local_debug', action="store_true", help="启用本地调试模式，不依赖线上环境")
parser.add_argument('--debug_batch_size', type=int, default=32, help="本地调试时的批大小")
args = parser.parse_known_args()[0]
# 导入必要的库
import os
import logging
import argparse
import functools
import contextlib
senmantic_id_token_num = 7
rag_seq_len = 7
long_term_seq_len = 512
token_rag_size = 64
item_rag_size = token_rag_size * senmantic_id_token_num
item_rag_magic_num = 30000
short_term_list_seq_len = 20
# 在本地调试模式下，首先导入TensorFlow
# python reco_llada.py --mode train --local_debug --with_kai_v2 --debug_batch_size 10  
if args.local_debug:
    import tensorflow as tf
    # 确保使用TensorFlow 1.x的API
    if hasattr(tf, 'disable_v2_behavior'):
        tf.disable_v2_behavior()
    else:
        tf.compat.v1.disable_eager_execution()
        tf = tf.compat.v1
    # 存储全局模拟张量的字典
    mock_tensors = {}
    
    # 定义一个函数，用于创建所有必要的模拟张量
    def create_mock_tensors(batch_size=32):
        """创建所有必要的模拟张量，作为全局变量供原始代码使用"""
        global mock_tensors
        
        print("[本地调试] 创建模拟张量...")
        
        # 存储所有创建的张量
        tensors = {}
        
        # 定义一些常用的维度
        dims = {
            'dim64': 64,
            'dim32': 32,
            'dim16': 16,
            'dim8': 8,
            'dim4': 4,
            'dim1': 1
        }
        
        # 创建用户相关的模拟embedding
        with tf.variable_scope("user_mock_tensors"):
            tensors['user_embedding_64'] = tf.random.normal([batch_size, 10 * dims['dim64']], 
                                                         mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                         name="user_embedding_64")
            tensors['user_embedding_8'] = tf.random.normal([batch_size, 8 * dims['dim8']], 
                                                        mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                        name="user_embedding_8")
            tensors['user_user_lhuc'] = tf.random.normal([batch_size, dims['dim32']], 
                                                      mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                      name="user_user_lhuc")
            tensors['user_mcp_user'] = tf.random.normal([batch_size, 2 * dims['dim64']], 
                                                     mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                     name="user_mcp_user")
            
            # 创建历史行为序列相关的模拟张量
            short_term_list_seq_len = 20
            tensors['user_short_term_pids'] = tf.random.normal([batch_size, short_term_list_seq_len, dims['dim64']], 
                                                            mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                            name="user_short_term_pids")
            tensors['user_short_term_aids'] = tf.random.normal([batch_size, short_term_list_seq_len, dims['dim64']], 
                                                            mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                            name="user_short_term_aids")
            tensors['user_short_term_tags'] = tf.random.normal([batch_size, short_term_list_seq_len, dims['dim8']], 
                                                            mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                            name="user_short_term_tags")
            tensors['user_short_term_times'] = tf.random.normal([batch_size, short_term_list_seq_len, dims['dim8']], 
                                                             mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                             name="user_short_term_times")
            tensors['user_short_term_play'] = tf.random.normal([batch_size, short_term_list_seq_len, dims['dim8']], 
                                                            mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                            name="user_short_term_play")
                                                         
        # 创建物品相关的模拟embedding
        with tf.variable_scope("item_mock_tensors"):
            tensors['item_embedding_64'] = tf.random.normal([batch_size, 4 * dims['dim64']], 
                                                         mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                         name="item_embedding_64")
            tensors['item_embedding_8'] = tf.random.normal([batch_size, 4 * dims['dim8']], 
                                                        mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                        name="item_embedding_8")
            tensors['item_user_lhuc'] = tf.random.normal([batch_size, dims['dim32']], 
                                                      mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                      name="item_user_lhuc")
            tensors['item_vtr_bias'] = tf.random.normal([batch_size, dims['dim8']], 
                                                      mean=0.0, stddev=0.1, dtype=tf.float32, 
                                                      name="item_vtr_bias")                                          
        # 更多可能需要的张量可以在这里添加...
        
        # 保存创建的模拟张量到全局字典
        mock_tensors = tensors
        
        print(f"[本地调试] 创建了 {len(tensors)} 个模拟张量用于调试")
        return tensors
        
    # 创建一个模拟的Kai config类，主要拦截embedding查询的创建
    class MockKaiConfig:
        def __init__(self):
            self.nn = MockNN()
            self.optimizer = MockOptimizer()
            self.GraphKeys = MockGraphKeys()
            # 记录当前批次大小
            self.batch_size = args.debug_batch_size
            # 记录所有新建的embedding
            self.embeddings = {}
            # 运行时配置
            self._runtime_option = type('RuntimeOption', (), {'mode': 'predict'})
            
        def Config(self):
            """返回配置实例自身，用于访问runtime_option"""
            self.runtime_option = self._runtime_option
            return self
            
        def get_dense_fea(self, name, dim=1, dtype=tf.float32):
            """模拟获取密集特征"""
            print(f"[MockKai] 获取密集特征: {name}, dim={dim}, dtype={dtype}")
            # 创建随机tensor来模拟特征
            shape = [self.batch_size, dim]
            with tf.variable_scope(f"feature_{name}"):
                if dtype == tf.int64 or dtype == tf.int32:
                    random_feature = tf.random.uniform(shape, minval=0, maxval=6, dtype=dtype, name=name)
                else:
                    random_feature = tf.random.normal(shape, mean=0.0, stddev=0.1, dtype=dtype, name=name)
            
            # 将创建的特征添加到全局模拟张量中
            global mock_tensors
            mock_tensors[name] = random_feature
            
            return random_feature
    
        def get_sparse_fea(self, name, dim=1, dtype=tf.int64):
            """模拟获取稀疏特征"""
            print(f"[MockKai] 获取稀疏特征: {name}, dim={dim}, dtype={dtype}")
            # 创建随机tensor来模拟稀疏特征
            shape = [self.batch_size, dim]
            with tf.variable_scope(f"sparse_feature_{name}"):
                # 对于稀疏特征,我们生成一些随机的ID
                random_feature = tf.random.uniform(shape, minval=0, maxval=1000, dtype=dtype, name=name)
            
            # 将创建的特征添加到全局模拟张量中
            global mock_tensors
            mock_tensors[f"sparse_{name}"] = random_feature
            
            # 返回元组,模拟原始接口的返回值
            return random_feature, None, None, None
    
        def get_step(self):
            """模拟获取step"""
            return 0
            
        def get_label(self, name):
            """模拟获取标签"""
            print(f"[MockKai] 获取标签: {name}")
            # 创建随机tensor来模拟标签
            shape = [self.batch_size, 1]
            with tf.variable_scope(f"label_{name}"):
                random_label = tf.random.uniform(shape, minval=0, maxval=2, dtype=tf.float32, name=name)
            
            # 将创建的标签添加到全局模拟张量中
            global mock_tensors
            mock_tensors[name] = random_label
            
            return random_label
            
        def new_embedding(self, name, dim, slots, **kwargs):
            """模拟创建embedding的函数"""
            print(f"[MockKai] 获取embedding: {name}, dim={dim}, slots={slots}")
            
            # 从全局mock_tensors中获取已经创建好的embedding
            global mock_tensors
            if name in mock_tensors:
                print(f"[MockKai] 使用已存在的embedding: {name}")
                embedding = mock_tensors[name]
                self.embeddings[name] = embedding
                return embedding
            
            # 如果在mock_tensors中找不到对应embedding，则创建一个警告并创建新的随机tensor
            if "expand" in kwargs:
                shape = [self.batch_size, kwargs["expand"], dim]  # 假设批次大小为self.batch_size
            else:
                shape = [self.batch_size, dim]  # 假设批次大小为self.batch_size
            print(f"[MockKai] 警告：在mock_tensors中未找到embedding '{name}'，创建新的随机tensor, shape={shape}")
            with tf.variable_scope(f"embedding_{name}"):
                random_embedding = tf.random.normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32, name=name)
            self.embeddings[name] = random_embedding
            
            # 将新创建的embedding添加到全局模拟张量中
            mock_tensors[name] = random_embedding
            
            return random_embedding
        
        def new_jit_context(self):
            """模拟JIT上下文"""
            @contextlib.contextmanager
            def dummy_context():
                yield
            return dummy_context()
            
        def declare_reallocate_slots(self, *args, **kwargs):
            """模拟slot重新分配"""
            return
        
        def set_slot_param_attr(self, *args, **kwargs):
            """模拟设置slot参数属性"""
            return
        
        def get_dense_trainable_variables(self):
            """获取可训练的密集变量"""
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        def get_collection(self, *args):
            """获取集合"""
            if args[0] == self.GraphKeys.EMBEDDING_INPUT:
                return self.embeddings.values()
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        def mock_and_profile(self, *args, **kwargs):
            """模拟性能分析"""
            return
        
        def build_model(self, *args, **kwargs):
            """构建模型"""
            return
            
        def dump_training_config(self, *args, **kwargs):
            """导出训练配置"""
            print("[MockKai] 导出训练配置")
            return
        
        def dump_predict_config(self, *args, **kwargs):
            """导出预测配置"""
            print("[MockKai] 导出预测配置")
            return
            
        def batch_norm(self, input, name, forward_with_moving_val=None, use_tf_cond=None, bn_decay=0.999, epsilon=1e-5, **kwargs):
            """模拟batch_norm操作"""
            print(f"[MockKai] 调用batch_norm: {name}")
            
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                # 使用TensorFlow的原生batch_normalization
                return tf.layers.batch_normalization(
                    inputs=input,
                    momentum=bn_decay,
                    epsilon=epsilon,
                    center=True,
                    scale=True,
                    training=(not forward_with_moving_val) if forward_with_moving_val is not None else None,
                    name=name
                )
                
        def mem_eff_attn(self, query, key, value, bias=None, scale_fp16=False, bf16=False, 
                         q_seqinfo=None, k_seqinfo=None, custom_mask_type=None, use_v2=False, **kwargs):
            """模拟内存高效的注意力机制"""
            print(f"[MockKai] 调用mem_eff_attn")
            
            # 提取形状信息
            batch_size = tf.shape(query)[0]
            q_seq_len = tf.shape(query)[1]
            num_heads = tf.shape(query)[2]
            head_dim = tf.shape(query)[3]
            
            # 重塑以便进行多头注意力计算
            q = tf.reshape(query, [batch_size, q_seq_len, num_heads, head_dim])
            k = tf.reshape(key, [batch_size, -1, num_heads, head_dim])
            v = tf.reshape(value, [batch_size, -1, num_heads, head_dim])
            
            # 转置以使头维度位于正确的位置
            q = tf.transpose(q, [0, 2, 1, 3])  # [batch, num_heads, q_seq_len, head_dim]
            k = tf.transpose(k, [0, 2, 1, 3])  # [batch, num_heads, kv_seq_len, head_dim]
            v = tf.transpose(v, [0, 2, 1, 3])  # [batch, num_heads, kv_seq_len, head_dim]
            
            # 计算注意力得分
            scores = tf.matmul(q, k, transpose_b=True)  # [batch, num_heads, q_seq_len, kv_seq_len]
            
            # 缩放因子
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scores = scores / tf.sqrt(dk)
            
            # 应用掩码（如果有）
            if custom_mask_type == "BlockDiagonalMask" and q_seqinfo is not None and k_seqinfo is not None:
                # 模拟块对角线掩码
                # 这里简化处理，不做实际掩码
                pass
            elif bias is not None:
                scores = scores + bias
                
            # 应用softmax获取注意力权重
            attention_weights = tf.nn.softmax(scores, axis=-1)
            
            # 应用注意力权重到值
            output = tf.matmul(attention_weights, v)  # [batch, num_heads, q_seq_len, head_dim]
            
            # 重塑回原始形状
            output = tf.transpose(output, [0, 2, 1, 3])  # [batch, q_seq_len, num_heads, head_dim]
            output_shape = tf.shape(query)
            output = tf.reshape(output, output_shape)
            
            return output
    class MockNN:
        def __init__(self):
            self.ParamAttr = MockParamAttr
            self.ProbabilityAccess = lambda x: f"ProbabilityAccess({x})"
            self.UnseendaysRecycle = lambda x, y: f"UnseendaysRecycle({x}, {y})"
            # 添加CustomMaskType枚举
            self.CustomMaskType = type('CustomMaskType', (), {
                'BlockDiagonalMask': 'BlockDiagonalMask',
                'BlockDiagonalCausalMask': 'BlockDiagonalCausalMask',
                'FullMask': 'FullMask'
            })
    class MockParamAttr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class MockOptimizer:
        def __init__(self):
            pass
        
        def Adam(self, lr):
            return tf.train.AdamOptimizer(learning_rate=lr)
        
        def AdamW(self, learning_rate, weight_decay):
            # TF1.x没有内置的AdamW，这里使用普通Adam替代
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
    class MockGraphKeys:
        def __init__(self):
            self.EMBEDDING_INPUT = "EMBEDDING_INPUT"
if args.with_kai_v2 and not args.local_debug:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    from kconf.get_config import get_double_config
    reco_channel = [
        'reco.reco_log_fine_sort_feature',
        'reco.reco_log_fine_sort_feature_exp',
        'reco.reco_log_fine_sort_feature_exp2',
        'reco.reco_log_fine_sort_feature_exp3',
        'reco.reco_log_fine_sort_feature_exp4',
        'reco.reco_log_fine_sort_feature_full_batch'
    ]
    reco_channel = None
    dcn_input_slots = [704, 704, 704, 1083, 1084, 704, 1083, 1084, 506, 704] + [26, 128]
    dcn_output_slots = [702, 706, 708, 1085, 1086, 709, 1087, 1088, 507, 710] + [791, 792]
    config.declare_reallocate_slots(dcn_input_slots, dcn_output_slots,
                                    remap=True, data_source_name=['train', 'test'],
                                    channel_name=reco_channel)
    GSU_MAGIC_NUM = 18000
    gsu_output_slots = [
        337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
        696, 700,
        1901, 1902, 1905, 1906, 1909]
    gsu_input_slots = [GSU_MAGIC_NUM + slot for slot in gsu_output_slots]
    config.declare_reallocate_slots(gsu_input_slots, gsu_output_slots,
                                    remap=False, data_source_name=['train', 'test'],
                                    channel_name=reco_channel)
    config.declare_reallocate_slots(
        [346 + GSU_MAGIC_NUM], [889], remap=True, data_source_name=['train', 'test'],
        channel_name=reco_channel)
    MILLION_GSU_MAGIC_NUM = 21000
    million_gsu_output_slots = [
        446, 444, 448, 447, 445, 450]
    million_gsu_input_slots = [
        MILLION_GSU_MAGIC_NUM + slot for slot in million_gsu_output_slots]
    config.declare_reallocate_slots(million_gsu_input_slots, million_gsu_output_slots,
                                    remap=False, data_source_name=['train', 'test'],
                                    channel_name=reco_channel)
    # user_geo_hash_slots = [1500, 1501, 1502, 1503, 1504, 1505]
    # config.declare_reallocate_slots(user_geo_hash_slots,
    #                                 [30 + slot for slot in user_geo_hash_slots],
    #                                 remap=True, data_source_name=['train', 'test'],
    #                                 channel_name=reco_channel)
    user_profile_slots = [1079, 184, 747, 752, 182, 757]
    config.declare_reallocate_slots(user_profile_slots,
                                    [1000 + slot for slot in user_profile_slots],
                                    remap=True, data_source_name=['train', 'test'],
                                    channel_name=reco_channel)
    config.declare_reallocate_slots(
        [704, 704] + [704, 38, 34, 26, 128] + [807, 817, 827] + [807, 817, 827],
        [904, 1395] + [707, 566, 568, 526, 546] + [867, 877, 887] + [907, 917, 927],
        remap=True, data_source_name=['train', 'test'],
        channel_name=reco_channel)
    use_flash_attention = False if config.Config().runtime_option.mode == "train" else False
    # for act 1w 64*512 for million 1K interest embedding
    user_id_or_device_id = config.get_dense_fea("user_id_or_device_id", 1, dtype=tf.int64)
    llsid = config.get_dense_fea("llsid", dim=1, dtype=tf.int64)
    photo_id = config.get_dense_fea("photo_id", dim=1, dtype=tf.int64)
    time_ms = config.get_dense_fea("time_ms", 1, dtype=tf.int64)
    user_id = config.get_dense_fea("user_id", 1, dtype=tf.int64)
    
    # remove: "1520"，"93", 
   
    item_rag_output_slots = ["1606", "1607", "26", "128", "71", "141", 
                             "142", "143", "417", "430", "776", "777", "778", 
                             "779", "780", "781", "782"] + ["6677"]
    item_rag_output_slots = [str(item_rag_magic_num + int(slot)) for slot in item_rag_output_slots]
    gsu_output_slots = ["1001", "1002", "1004", "1006", "1007", "1008", "1009", "1010", "1011", "1013", "1014"]
    
    config.declare_remote_gsu(
        kess_service = "grpc_item_rag_user_longterm",
        request_type = "ntp_train_request_7x64",
        llsid_attr = "user_id_or_device_id",
        uid_attr="user_id",
        pid_attr="photo_id",
        input_attrs=["time_ms", "user_id"],
        input_common_flags= [False, True],
        output_attrs=["tokens", "colossus_time_s"] + item_rag_output_slots + gsu_output_slots,
        output_column_types=["list<int64>", "list<int64>"] + ["list<int64>"] * len(item_rag_output_slots) + ["list<int64>"] * len(gsu_output_slots),
        output_common_flags=[False, False] + [False] * len(item_rag_output_slots) + [False] * len(gsu_output_slots),
        data_source_name="train",
        timeout_ms=60000)
    def filter_mask_wrapper(dataset):
        dataset.add_feature('llsid', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('stid_mix_filter_flag', dataset.DENSE, tf.int64, 1)
        def mask_fn(batch):
            sample_llsid = batch['llsid']
            # sample_ratio = get_double_config("reco.model2.reco_llada_sample_filter_exp1")
            sample_ratio = 0.04
            hash_val = tf.cast(tf.bitwise.bitwise_and(sample_llsid, 0xffff), tf.float32)
            hash_ratio = hash_val / tf.cast(0xffff, tf.float32)
            sample_mask = tf.cast(tf.greater(hash_ratio, sample_ratio), tf.bool)
            stid_mix_filter_flag = batch['stid_mix_filter_flag']
            stid_mix_mask = tf.cast(tf.equal(stid_mix_filter_flag, 1), tf.bool) # [b, 1]
            mask = tf.logical_or(sample_mask, stid_mix_mask)
            return mask
        
        return mask_fn
    
    config.declare_sample_filter(
        filter_mask_wrapper, data_source_name=['train', 'test'], channel_name=reco_channel)
elif args.local_debug:
    # 在本地调试模式下使用模拟环境
    print("启用本地调试模式，使用TensorFlow但模拟环境")
    
    # 无论是否启用了with_kai_v2，都使用模拟配置
    config = MockKaiConfig()
    
    # 为了保持与原代码一致，也定义一些常量
    GSU_MAGIC_NUM = 18000
    MILLION_GSU_MAGIC_NUM = 21000
    reco_channel = None
    use_flash_attention = False
    
    # 如果不是使用kai_v2，还需要模拟MioConfig相关的类和函数
    if not args.with_kai_v2:
        print("[本地调试] 同时模拟MioConfig环境")
        # 定义模拟的MioConfig类
        class MockMioConfig:
            @staticmethod
            def from_base_yaml(*args, **kwargs):
                return MockMioConfig()
                
            def get_dense_fea(self, *args, **kwargs):
                return None
                
            def dump_training_config(self, *args, **kwargs):
                print("[本地调试] 调用dump_training_config")
                return
                
        # 定义模拟的MioVariable类
        class MockMioVariable:
            pass
            
        # 定义模拟的MioEmbedding类
        class MockMioEmbedding:
            pass
            
        # 将模拟类注入到全局命名空间
        MioConfig = MockMioConfig
        MioVariable = MockMioVariable
        MioEmbedding = MockMioEmbedding
    
    # 注入全局变量来模拟embedding查询结果
    print("[本地调试] 初始化模拟张量...")
    mock_tensor_dict = create_mock_tensors(batch_size=args.debug_batch_size)
    
    # 将mock_tensors中的所有张量注入到全局命名空间
    for name, tensor in mock_tensor_dict.items():
        globals()[name] = tensor
        print(f"[本地调试] 注入全局模拟张量: {name}, 形状: {tensor.get_shape()}")
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    from mio_tensorflow.variable import MioVariable, MioEmbedding
    if not args.dryrun:
        # monkey patch
        import mio_tensorflow.patch as mio_tensorflow_patch
        mio_tensorflow_patch.apply()
    import tensorflow as tf
    from tensorflow.keras.backend import expand_dims, repeat_elements, mean
    from mio_tensorflow.config import MioConfig
    from mio_tensorflow.variable import MioVariable, MioEmbedding
    logging.basicConfig()
    base_config = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "./dnn-plugin.yaml.bak")
    if args.mode == "predict":
        config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                        dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                        predict=True)
        use_flash_attention = False
    else:
        config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                        dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False)
        use_flash_attention = False
def swish(x, beta=1.0):
    return x * tf.nn.sigmoid(beta * x)
def conditional_tf_print(condition_func, *args, **kwargs):
    return tf.cond(
        condition_func(),
        lambda: tf.print(*args, **kwargs),
        lambda: tf.no_op(),
    )
def print_by_step(step, *args, **kwargs):
    train_step = config.get_step()
    return tf.cond(
        tf.equal(tf.mod(train_step, step), 0),
        lambda: tf.print(*args, **kwargs),
        lambda: tf.no_op(),
    )
use_matmul_ex = False
use_xla = False if args.mode == "predict" else True
# use_xla = False
dense_norm_type = 'batch_norm'
gate_norm_type = 'batch_norm'
# emb_norm_type = 'batch_norm'
emb_norm_type = None
variable_norm_type = 'batch_norm'
bottom_norm_type = 'batch_norm'
trans_norm_type = 'layer_norm'
sparse_norm_type = None
# dense_activation = tf.nn.relu
dense_activation = swish
sparse_tower_units = [128, 64]
dense_tower_units = [256, 128]
def new_xla_jit_context():
    if args.local_debug:
        # 本地调试模式下返回空上下文管理器
        @contextlib.contextmanager
        def dummy_context():
            yield
        return dummy_context()
    else:
        # 恢复原始逻辑
        if use_xla:
            return tf.xla.experimental.jit_scope()
        else:
            return contextlib.suppress()
def tf_name_scope(scope_name):
    if args.local_debug:
        # 在本地调试模式下使用tf的name_scope
        return tf.name_scope(scope_name)
    else:
        # 恢复原始逻辑
        if args.mode == "predict":
            return tf.name_scope(None)
        else:
            return tf.name_scope(scope_name)
def P_new(tensor, name=None, summarize=17, **kwargs):
    if args.mode == "train":
        return tf.Print(tensor, [tf.shape(tensor), tensor], tensor.name if name is None else name, summarize=summarize,
                        **kwargs)
    else:
        return tensor
def P(tensor, name=None, summarize=17, enable_print_in_training=False, **kwargs):
    if enable_print_in_training and args.mode == "train":
        return tf.Print(tensor, [tf.shape(tensor), tensor], tensor.name if name is None else name, summarize=summarize,
                        **kwargs)
    else:
        return tensor
def P_l(list_of_tensors, name=None, **kwargs):
    return [P(tensor, name.format(i), **kwargs) for i, tensor in enumerate(list_of_tensors)]
def slice_inputs(inputs):
    if isinstance(inputs, list):
        return inputs[0], inputs[1:]
    else:
        return inputs, []
global_fp16_regularizers = []
enable_fp16_regularizer = False
def fp16_regularizer(f):
    fp16_half_threshold = 65504 / 2
    def g(*args, **kwargs):
        ret = f(*args, **kwargs)
        if enable_fp16_regularizer:
            # L1 reg
            l1reg = tf.reduce_mean(tf.maximum(tf.math.abs(
                ret), fp16_half_threshold) - fp16_half_threshold)
            # tf.summary.scalar(f'l1reg_{ret.name}', l1reg)
            # L2 reg
            # threshold = tf.ones_like(ret) * fp16_half_threshold
            # l2reg = tf.losses.mean_squared_error(tf.maximum(tf.math.abs(ret), threshold), threshold)
            # tf.summary.scalar(f'l2reg_{ret.name}', l2reg)
            global_fp16_regularizers.append(l1reg)
        return ret
    return g
def matmul_with_cast(matmul_func=tf.matmul, in_type=tf.float32,
                     is_tf1=True):
    """
    in_type can be float, bf16, fp16
    output_type is expected to be float
    """
    def func(a, b, **kwargs):
        if a.dtype != in_type:
            a = tf.cast(a, in_type)
        if b.dtype != in_type:
            b = tf.cast(b, in_type)
        if is_tf1 and matmul_func == tf.matmul_cast:
            out_type = in_type
            c = matmul_func(a, b, DstT=out_type, **kwargs)
        else:
            c = matmul_func(a, b, **kwargs)
        if c.dtype != tf.float32:
            c = tf.cast(c, tf.float32)
        return c
    return func
@fp16_regularizer
def matmul_ex_wrapper(a, b, transpose_a=False, transpose_b=False):
#    return tf.matmul_float_to_bf16(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    return  matmul_with_cast(
        matmul_func=tf.matmul, in_type=tf.bfloat16, is_tf1=False)(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
@fp16_regularizer
def matmul_wrapper(*args, **kwargs):
    return tf.matmul(*args, **kwargs)
r = fp16_regularizer(lambda x, name: tf.identity(x, name=name))
matmul_fun = matmul_ex_wrapper if use_matmul_ex else matmul_wrapper
def emb_norm(inputs, scope_name):
    return norm(inputs, norm_type=emb_norm_type, scope_name=scope_name)
def norm(x, begin_axis=-1, eps=1e-5, norm_type=dense_norm_type, scope_name=''):
    """Normalization layer."""
    shape = x.shape.as_list()
    axes = list(range(len(shape)))[begin_axis:]
    if norm_type == 'layer_norm':
        mean, var = tf.nn.moments(x, axes, keepdims=True, name=f"{scope_name}_moments")
        x = (x - mean) * tf.math.rsqrt(var + eps)
        gamma = tf.get_variable(f'layer_norm_gamma_{scope_name}', shape=x.shape.as_list()[
                                begin_axis:], initializer=tf.initializers.ones())
        beta = tf.get_variable(f'layer_norm_beta_{scope_name}', shape=x.shape.as_list()[
                               begin_axis:], initializer=tf.initializers.zeros())
        output = gamma * x + beta
        return output
    elif norm_type == 'scale_norm':
        mean_square = tf.reduce_mean(tf.math.square(x), axes, keepdims=True)
        x = x * tf.rsqrt(mean_square + eps)
        scalar = tf.get_variable(f'scalar_{scope_name}', shape=(
        ), initializer=tf.constant_initializer(1.0))
        return scalar * x
    elif norm_type == 'batch_norm':
        if args.local_debug:
            # 在本地调试模式下使用前面实现的MockKaiConfig.batch_norm
            output = config.batch_norm(
                input=x, 
                name=f'batch_norm_{scope_name}',
                bn_decay=0.0 if args.mode != "predict" else 0.999
            )
        elif args.mode == "predict":
            output = config.batch_norm(input=x, name=f'batch_norm_{scope_name}')
        else:
            output = config.batch_norm(input=x, name=f'batch_norm_{scope_name}',
                                        #forward_with_moving_val=False,
                                        #use_tf_cond=False,
                                        bn_decay=0.0)
        return output
    elif norm_type is None:
        return x
def mio_dense_layer(inputs, units, activation, name, weight_name, norm_type=None, bias=True):
    """
    tf.Dense-like layer similar to that of mio-dnn
    """
    i, extra_i = slice_inputs(inputs)
    with tf_name_scope(name):
        # o = i @ tf.get_variable(weight_name, (i.get_shape()[1], units))
        o = matmul_fun(i, tf.get_variable(
            weight_name, (i.get_shape()[-1], units)))
        for idx, extra_i in enumerate(extra_i):
            # o += extra_i @ tf.get_variable(f"{weight_name}_extra_{idx}", (extra_i.get_shape()[1], units))
            o += matmul_fun(extra_i, tf.get_variable(
                f"{weight_name}_extra_{idx}", (extra_i.get_shape()[-1], units)))
            
        if bias:
            bias = tf.get_variable(f"{weight_name}_bias", (units))
            o = tf.nn.bias_add(o, bias)
        if norm_type is not None:
            print("mio_dense_layer: name: {0}; norm_type: {1}".format(name, norm_type))
            o = norm(o, norm_type=norm_type, scope_name=weight_name)
        if activation is not None:
            o = activation(o)
        return o
def simple_lhuc_network(inputs, unit1, unit2, name, weight_name, norm_type=None):
    with tf_name_scope(f"{name}_lhuc"):
        output = [emb_norm(tf.concat(inputs, 1), scope_name=weight_name)]
        with tf_name_scope(f"{name}_lhuc_layer_0"):
            output = mio_dense_layer(output, unit1, dense_activation, name=f"dense_{name}_0",
                                     weight_name=f"{weight_name}_layer1_param", norm_type=norm_type)
        with tf_name_scope(f"{name}_lhuc_layer_1"):
            output = 2.0 * mio_dense_layer(output, unit2, tf.nn.sigmoid, name=f"dense_{name}_1",
                                           weight_name=f"{weight_name}_layer2_param",
                                           norm_type=norm_type)
        return output
###################### user embedding ###########################
compress_kwargs = {}
if args.mode in ["predict", "gsu"]:
    compress_kwargs["compress_group"] = "USER"
############# input embedding prepare #############
### user_embedding
user_embedding_64 = config.new_embedding("user_embedding_64", dim=64, slots=[34, 38, 216, 217, 218, 219, 220],
                                         **compress_kwargs)
user_embedding_8 = config.new_embedding("user_embedding_8", dim=8, slots=[33, 145, 180, 181, 189, 600],
                                        **compress_kwargs)
user_user_lhuc = config.new_embedding("user_lhuc", dim=32, slots=[
                                      424, 703], **compress_kwargs)
user_mcp_user = config.new_embedding("user_mcp_user", dim=64, slots=[
                                     748, 749], **compress_kwargs)
user_evtr_ids = config.new_embedding(
    "user_evtr_ids", dim=64, slots=[827], **compress_kwargs)
user_clevtr_ids = config.new_embedding(
    "user_clevtr_ids", dim=64, slots=[927], **compress_kwargs)
user_peak_evtr_ids = config.new_embedding(
    "user_peak_evtr_ids", dim=64, slots=[887], **compress_kwargs)
user_ptr_ids = config.new_embedding(
    "user_ptr_ids", dim=64, slots=[222], **compress_kwargs)
user_wtr_ids = config.new_embedding(
    "wtr_ids", dim=64, slots=[499], **compress_kwargs)
user_epstr_ids = config.new_embedding("user_epstr_ids", dim=32, slots=[
                                      830], **compress_kwargs)  # id
user_ltr_ids = config.new_embedding(
    "user_ltr_ids", dim=32, slots=[831], **compress_kwargs)
user_ftr_ids = config.new_embedding(
    "user_ftr_ids", dim=32, slots=[832], **compress_kwargs)
user_dtr_ids = config.new_embedding(
    "user_dtr_ids", dim=32, slots=[834], **compress_kwargs)
user_lvtr_ids = config.new_embedding(
    "user_lvtr_ids", dim=32, slots=[420], **compress_kwargs)
user_svr_ids = config.new_embedding(
    "user_svr_ids", dim=32, slots=[722], **compress_kwargs)
user_live_ids = config.new_embedding("user_live_ids", dim=32, slots=[
                                     367], **compress_kwargs)  # id
user_cpr_ids = config.new_embedding("user_cpr_ids", dim=32, slots=[
                                    738], **compress_kwargs)  # id
user_dfvtr_ids = config.new_embedding(
    "user_dfvtr_ids", dim=16, slots=[828], **compress_kwargs)
user_thanos_etcm_ids = config.new_embedding(
    "user_thanos_etcm_ids", dim=16, slots=[829], **compress_kwargs)  # id
user_cmtr_ids = config.new_embedding(
    "user_cmtr_ids", dim=16, slots=[833], **compress_kwargs)
user_cmef_ids = config.new_embedding(
    "user_cmef_ids", dim=16, slots=[836], **compress_kwargs)
user_hot_lvtr_ids = config.new_embedding(
    "hot_lvtr_ids", dim=16, slots=[386], **compress_kwargs)
user_profile_time_cnt_id = config.new_embedding(
    "user_profile_time_cnt_id", dim=16, slots=[1024], **compress_kwargs)
user_watch_time_ids = config.new_embedding(
    "user_watch_time_ids", dim=16, slots=[495], **compress_kwargs)
user_watch_time_ids2 = config.new_embedding(
    "user_watch_time_ids2", dim=16, slots=[366], **compress_kwargs)
user_mmu_ids = config.new_embedding("user_mmu_ids", dim=32, slots=[
                                    839], **compress_kwargs)  # id
user_fountain_evtr_v2_ids = config.new_embedding("user_fountain_evtr_v2_ids", dim=64, slots=[559],
                                                 **compress_kwargs)  # id
user_ctr_input_64 = config.new_embedding("user_ctr_input_64", dim=64, slots=[
                                         282, 287, 336, 1140], **compress_kwargs)
user_ctr_input_8 = config.new_embedding("user_ctr_input_8", dim=8, slots=[
                                        756, 753], **compress_kwargs)
user_ctr_ids = config.new_embedding(
    "user_ctr_ids", dim=64, slots=[557], **compress_kwargs)
user_ctr_lhuc = config.new_embedding(
    "user_ctr_lhuc", dim=64, slots=[558], **compress_kwargs)
user_ctr_lhuc_v1 = config.new_embedding(
    "user_ctr_lhuc_v1", dim=64, slots=[835], **compress_kwargs)
user_fountain_enter_ids = config.new_embedding(
    "user_fountain_enter_ids", dim=64, slots=[563], **compress_kwargs)
user_fountain_effective_consumer_ids = config.new_embedding(
    "user_fountain_effective_consumer_ids", dim=64, slots=[565], **compress_kwargs)
user_wtd_v2_bias = config.new_embedding("user_wtd_v2_bias", dim=8, slots=[
                                        706], **compress_kwargs)  # tabid
user_wtd_v2_ids = config.new_embedding("user_wtd_v2_ids", dim=64, slots=[
                                       562], **compress_kwargs)   # id
user_adp_wtd_bias = config.new_embedding("user_adp_wtd_bias", dim=8, slots=[
                                        707], **compress_kwargs)  # tabid
user_adp_wtd_ids = config.new_embedding("user_adp_wtd_ids", dim=64, slots=[
                                       566, 568], **compress_kwargs)   # id
user_mix_tabinfo = config.new_embedding("mix_tabinfo", dim=8, slots=[
                                        702], **compress_kwargs)  # tabid
user_mix_ids = config.new_embedding("user_mix_ids", dim=32, slots=[
                                    560], **compress_kwargs)    # id
user_wtd_evtr_tabinfo = config.new_embedding(
    "user_wtd_evtr_tabinfo", dim=8, slots=[708], **compress_kwargs)  # tabid
user_wtd_lvtr_tabinfo = config.new_embedding(
    "user_wtd_lvtr_tabinfo", dim=8, slots=[709], **compress_kwargs)  # tabid
user_pro_ids = config.new_embedding(
    "user_pro_ids", dim=32, slots=[564], **compress_kwargs)
user_pro_tabinfo = config.new_embedding(
    "pro_tabinfo", dim=8, slots=[710], **compress_kwargs)
user_l2r_ids = config.new_embedding(
    "user_l2r_ids", dim=32, slots=[1391], **compress_kwargs)
user_l2r_tabinfo = config.new_embedding(
    "user_l2r_tabinfo", dim=8, slots=[1395], **compress_kwargs)
user_local_life_city_id = config.new_embedding(
    "user_local_life_city_id", dim=8, slots=[1603], **compress_kwargs)
user_base_tab_id = config.new_embedding("user_base_tab_id", dim=64, slots=[704], **compress_kwargs)  # tabid
user_base_profile_feature = config.new_embedding("user_base_profile_feature", dim=64, slots=[
    2079, 1184, 1747, 1752, 1182, 1757], **compress_kwargs)  # age, gender, low_active, is_south, city, city3
user_short_term_pids = config.new_embedding("user_short_term_pids", dim=64, expand=short_term_list_seq_len,
                                            slots=[246], **compress_kwargs)
user_short_term_aids = config.new_embedding("user_short_term_aids", dim=64, expand=short_term_list_seq_len,
                                            slots=[247], **compress_kwargs)
user_short_term_tags = config.new_embedding("user_short_term_tags", dim=8, expand=short_term_list_seq_len,
                                            slots=[253], **compress_kwargs)
user_short_term_times = config.new_embedding("user_short_term_times", dim=8, expand=short_term_list_seq_len,
                                             slots=[250], **compress_kwargs)
user_short_term_play = config.new_embedding("user_short_term_play", dim=8, expand=short_term_list_seq_len,
                                            slots=[252], **compress_kwargs)
# if args.mode == "train":
#     with tf.xla.experimental.jit_scope(compile_ops=False):
#         item_long_term_pids3, user_id_or_device_id_index = config.new_embedding("item_long_term_pids3", dim=64, expand=long_term_seq_len, slots=[1001], compress_group="user_id_or_device_id")
#         item_long_term_aids3, user_id_or_device_id_index = config.new_embedding("item_long_term_aids3", dim=64, expand=long_term_seq_len, slots=[1002], compress_group="user_id_or_device_id")
#         item_long_term_label3, user_id_or_device_id_index = config.new_embedding("item_long_term_label3", dim=8, expand=long_term_seq_len, slots=[1004], compress_group="user_id_or_device_id")
#         item_long_term_tag3, user_id_or_device_id_index = config.new_embedding("item_long_term_tag3", dim=8, expand=long_term_seq_len, slots=[1006], compress_group="user_id_or_device_id")
#         item_long_term_duration3, user_id_or_device_id_index = config.new_embedding("item_long_term_duration3", dim=8, expand=long_term_seq_len, slots=[1007], compress_group="user_id_or_device_id")
#         item_long_term_play_time3, user_id_or_device_id_index = config.new_embedding("item_long_term_play_time3", dim=8, expand=long_term_seq_len, slots=[1008], compress_group="user_id_or_device_id")
#         item_long_term_channel3, user_id_or_device_id_index = config.new_embedding("item_long_term_channel3", dim=8, expand=long_term_seq_len, slots=[1009], compress_group="user_id_or_device_id")
#         item_long_term_play_x_duration3, user_id_or_device_id_index = config.new_embedding("item_long_term_play_x_duration3", dim=8, expand=long_term_seq_len, slots=[1010], compress_group="user_id_or_device_id")
#         item_long_term_day_diff3, user_id_or_device_id_index = config.new_embedding("item_long_term_day_diff3", dim=8, expand=long_term_seq_len, slots=[1011], compress_group="user_id_or_device_id")
#         item_long_term_mindiff3, user_id_or_device_id_index = config.new_embedding("item_long_term_mindiff3", dim=8, expand=long_term_seq_len, slots=[1013], compress_group="user_id_or_device_id")
#         item_long_term_abspose3, user_id_or_device_id_index = config.new_embedding("item_long_term_abspose3", dim=8, expand=long_term_seq_len, slots=[1014], compress_group="user_id_or_device_id")
#         colossus_time_s, user_id_or_device_id_index = config.get_dense_fea("colossus_time_s", long_term_seq_len, dtype=tf.int64, compress_group="user_id_or_device_id")
# else:
#     item_long_term_pids3 = config.new_embedding("item_long_term_pids3", dim=64, expand=long_term_seq_len, slots=[1001])
#     item_long_term_aids3 = config.new_embedding("item_long_term_aids3", dim=64, expand=long_term_seq_len, slots=[1002])
#     item_long_term_label3 = config.new_embedding("item_long_term_label3", dim=8, expand=long_term_seq_len, slots=[1004])
#     item_long_term_tag3 = config.new_embedding("item_long_term_tag3", dim=8, expand=long_term_seq_len, slots=[1006])
#     item_long_term_duration3 = config.new_embedding("item_long_term_duration3", dim=8, expand=long_term_seq_len, slots=[1007])
#     item_long_term_play_time3 = config.new_embedding("item_long_term_play_time3", dim=8, expand=long_term_seq_len, slots=[1008])
#     item_long_term_channel3 = config.new_embedding("item_long_term_channel3", dim=8, expand=long_term_seq_len, slots=[1009])
#     item_long_term_play_x_duration3 = config.new_embedding("item_long_term_play_x_duration3", dim=8, expand=long_term_seq_len, slots=[1010])
#     item_long_term_day_diff3 = config.new_embedding("item_long_term_day_diff3", dim=8, expand=long_term_seq_len, slots=[1011])
#     item_long_term_mindiff3 = config.new_embedding("item_long_term_mindiff3", dim=8, expand=long_term_seq_len, slots=[1013])
#     item_long_term_abspose3 = config.new_embedding("item_long_term_abspose3", dim=8, expand=long_term_seq_len, slots=[1014])
#     colossus_time_s = config.get_dense_fea("colossus_time_s", long_term_seq_len, dtype=tf.int64)
item_long_term_pids3 = config.new_embedding("item_long_term_pids3", dim=64, expand=long_term_seq_len, slots=[1001])
item_long_term_aids3 = config.new_embedding("item_long_term_aids3", dim=64, expand=long_term_seq_len, slots=[1002])
item_long_term_label3 = config.new_embedding("item_long_term_label3", dim=8, expand=long_term_seq_len, slots=[1004])
item_long_term_tag3 = config.new_embedding("item_long_term_tag3", dim=8, expand=long_term_seq_len, slots=[1006])
item_long_term_duration3 = config.new_embedding("item_long_term_duration3", dim=8, expand=long_term_seq_len, slots=[1007])
item_long_term_play_time3 = config.new_embedding("item_long_term_play_time3", dim=8, expand=long_term_seq_len, slots=[1008])
item_long_term_channel3 = config.new_embedding("item_long_term_channel3", dim=8, expand=long_term_seq_len, slots=[1009])
item_long_term_play_x_duration3 = config.new_embedding("item_long_term_play_x_duration3", dim=8, expand=long_term_seq_len, slots=[1010])
item_long_term_day_diff3 = config.new_embedding("item_long_term_day_diff3", dim=8, expand=long_term_seq_len, slots=[1011])
item_long_term_mindiff3 = config.new_embedding("item_long_term_mindiff3", dim=8, expand=long_term_seq_len, slots=[1013])
item_long_term_abspose3 = config.new_embedding("item_long_term_abspose3", dim=8, expand=long_term_seq_len, slots=[1014])
colossus_time_s = config.get_dense_fea("colossus_time_s", long_term_seq_len, dtype=tf.int64)
### item RAG embedding
# item_rag_item_age_hour = config.new_embedding("item_rag_1520", dim=16, expand=item_rag_size, slots=[1520])
rag_item_local_life_id_1 = config.new_embedding("rag_item_local_life_id_1", dim=8, 
                                                expand=item_rag_size, slots=[item_rag_magic_num + 1606])
rag_item_local_life_id_2 = config.new_embedding("rag_item_local_life_id_2", dim=8, 
                                                expand=item_rag_size, slots=[item_rag_magic_num + 1607])
rag_item_local_life_emb = tf.concat([
    tf.reshape(rag_item_local_life_id_1, (-1, item_rag_size, 8)), 
    tf.reshape(rag_item_local_life_id_2, (-1, item_rag_size, 8))
], axis=-1)
rag_item_embedding_64_pid = config.new_embedding("rag_item_embedding_64_pid", dim=64, 
                                                 expand=item_rag_size, slots=[item_rag_magic_num + 26])
rag_item_embedding_64_aid = config.new_embedding("rag_item_embedding_64_aid", dim=64, 
                                                 expand=item_rag_size, slots=[item_rag_magic_num + 128])
rag_item_embedding_64 = tf.concat([
    tf.reshape(rag_item_embedding_64_pid, (-1, item_rag_size, 64)), 
    tf.reshape(rag_item_embedding_64_aid, (-1, item_rag_size, 64))
], axis=-1)
rag_item_embedding_8_list = []
rag_item_embedding_slot_list = [71, 141, 142, 143, 430, 417]
rag_item_embedding_slot_list = [item_rag_magic_num + slot for slot in rag_item_embedding_slot_list]
for slot_id in rag_item_embedding_slot_list:
    
    if slot_id == 93:
        expand_size = item_rag_size * 2
    elif slot_id == 418:
        expand_size = item_rag_size * 3
    else:
        expand_size = item_rag_size
    slot_embedding = config.new_embedding(f"rag_item_embedding_8_{slot_id}", dim=8, 
                            expand=expand_size, slots=[slot_id])
    
    if slot_id == 93:
        slot_embedding = tf.reshape(slot_embedding, (-1, item_rag_size, 2, 8))
        slot_embedding = tf.reduce_sum(slot_embedding, axis=2)
    elif slot_id == 418:
        slot_embedding = tf.reshape(slot_embedding, (-1, item_rag_size, 3, 8))
        slot_embedding = tf.reduce_sum(slot_embedding, axis=2)
    else:
        slot_embedding = tf.reshape(slot_embedding, (-1, expand_size, 8))
    rag_item_embedding_8_list.append(slot_embedding)
rag_item_embedding_8 = tf.concat(rag_item_embedding_8_list, axis=-1)
rag_item_nebula_stats_list = []
rag_item_nebula_stats_slot_list = [776, 777, 778, 779, 780, 781, 782]
rag_item_nebula_stats_slot_list = [item_rag_magic_num + slot for slot in rag_item_nebula_stats_slot_list]
for slot_id in rag_item_nebula_stats_slot_list:
    expand_size = item_rag_size
    slot_embedding = config.new_embedding(f"rag_item_nebula_stats_{slot_id}", dim=8, 
                            expand=expand_size, slots=[slot_id])
    slot_embedding = tf.reshape(slot_embedding, (-1, expand_size, 8))
    rag_item_nebula_stats_list.append(slot_embedding)
rag_item_nebula_stats = tf.concat(rag_item_nebula_stats_list, axis=-1)
rag_node_embedding = config.new_embedding("rag_node_embedding", dim=128, expand=item_rag_size, slots=[item_rag_magic_num + 6677])
rag_node_embedding = tf.reshape(rag_node_embedding, (-1, item_rag_size, 128))
if args.mode == "train":
    semantic_id_v2 = config.get_dense_fea("tokens", dim=senmantic_id_token_num, dtype=tf.int64)
    # rag_fea_ids, _, _, _ = config.get_sparse_fea('36677', dim=item_rag_size, dtype=tf.int64)
    # rag_fea_ids = P(rag_fea_ids, enable_print_in_training=True)
    # train_step = config.get_step()
    # print_debug = conditional_tf_print(
    #     lambda: tf.equal(tf.mod(train_step, 10), 1),
    #     "step: ", train_step,
    #     "semantic_id_v2:", semantic_id_v2,
    #     "rag_fea_ids:", rag_fea_ids,
    #     output_stream=sys.stdout,
    #     summarize=20,
    # )
    # with tf.control_dependencies([print_debug]):
    #     semantic_id_v2 = tf.identity(semantic_id_v2)
    
    labels_map = {
        "adp_effective_view_fix": config.get_dense_fea('adp_effective_view_fix', 1, dtype=tf.int64),
        "click": config.get_label("click"),
        "like": config.get_label("like"),
        "follow": config.get_label("follow"),
        "forward": config.get_label("forward"),
        "comment": config.get_label("comment"),
        "long_view": config.get_label("long_view"),
        "short_view": config.get_label("short_view"),
        "play_complete": config.get_label("play_complete")
    }
    semantic_id_v2 = tf.reshape(tf.cast(semantic_id_v2, tf.int32), (-1, senmantic_id_token_num))
    labels = [tf.reshape(tf.cast(labels_map[key], tf.float32), (-1, 1)) for key in labels_map]  
else:
    semantic_id_v2 = config.get_dense_fea("semantic_id_v2", dim=senmantic_id_token_num, dtype=tf.float32)
    labels = [
        config.get_dense_fea('adp_effective_view_fix', 1, dtype=tf.float32),
        config.get_dense_fea("click", 1, dtype=tf.float32),
        config.get_dense_fea("like", 1, dtype=tf.float32),
        config.get_dense_fea("follow", 1, dtype=tf.float32),
        config.get_dense_fea("forward", 1, dtype=tf.float32),
        config.get_dense_fea("comment", 1, dtype=tf.float32),
        config.get_dense_fea("long_view", 1, dtype=tf.float32),
        config.get_dense_fea("short_view", 1, dtype=tf.float32),
        config.get_dense_fea("play_complete", 1, dtype=tf.float32)
    ]
    semantic_id_v2 = tf.reshape(tf.cast(semantic_id_v2, tf.int32), (-1, senmantic_id_token_num))    
    labels = [tf.reshape(l, (-1, 1)) for l in labels]
### slots list
########### input embedding prepare end ###########
############# feature prepare #############
new_initializer = lambda *args, **kwargs: 0.01*tf.glorot_uniform_initializer()(*args, **kwargs)
def dense_proj_tokens(input, token_num, token_dim, name):
    # input: [b, concat_dim]
    # output: [b, token_num, token_dim]
    input = mio_dense_layer(input, token_dim * token_num, 
                            None, f"{name}_proj_tokens", f"{name}_proj_tokens_param", 
                            bias=False)
    input = tf.reshape(input, (-1, token_num, token_dim))
    return input
def token_proj_tokens(input, token_dim, name):
    # input: [b, seq_len, concat_dim]
    # output: [b, seq_len, token_dim]
    input = mio_dense_layer(input, token_dim, 
                            None, f"{name}_proj_tokens", f"{name}_proj_tokens_param",
                            bias=False)
    return input
def prepare_item_tokens(vocab_size, token_num, d_model):
    # build item token embedding
    batch_size = tf.shape(semantic_id_v2)[0]
    item_token_embedding = tf.get_variable("item_token_embedding", (vocab_size, d_model))
    item_embeddings = tf.nn.embedding_lookup(item_token_embedding, semantic_id_v2)
    
    if args.mode == "train":
        # build all zeros mask
        zeros_id_num = tf.reduce_sum(tf.cast(tf.equal(semantic_id_v2, 0), tf.int32), axis=-1) # [b]
        all_zeros_mask = tf.cast(tf.equal(zeros_id_num, token_num), tf.int32) # [b] 1 for all zeros
    else:
        all_zeros_mask = tf.zeros((batch_size, 1), dtype=tf.int32)
    
    return item_embeddings, all_zeros_mask
def prepare_query_input():
    if args.local_debug:
        def get_col(x): return int(x.get_shape()[1])
    else:
        def get_col(x): return x.get_shape()[1].value
    
    update_inputs_dim64 = [user_embedding_64, user_mcp_user]
    # update_inputs_dim64 = [user_embedding_64, item_embedding_64, 
    #                         user_mcp_user, item_mcp_photo]
    num_dim64_slots = sum(map(get_col, update_inputs_dim64)) // 64
    update_inputs_dim8 = [user_embedding_8]
    # update_inputs_dim8 = [user_embedding_8, item_embedding_8]
    num_dim8_slots = sum(map(get_col, update_inputs_dim8)) // 8
    slot_gate_input = [user_user_lhuc] + [tf.stop_gradient(embedding) for embedding in
                                            (update_inputs_dim64 + update_inputs_dim8)]
    slot_gate = simple_lhuc_network(slot_gate_input, 512, num_dim64_slots + num_dim8_slots, "slot_gate_new",
                                    "slot_gate_new")
    update_inputs_dim64_concat = tf.reshape(
        tf.concat(update_inputs_dim64, 1), (-1, num_dim64_slots, 64))  # [N, 13, 64]
    update_inputs_dim8_concat = tf.reshape(
        tf.concat(update_inputs_dim8, 1), (-1, num_dim8_slots, 8))  # [N, 17, 8]
    slot_gate_split = tf.split(slot_gate, [num_dim64_slots, num_dim8_slots], 1)
    inputs_slot_gated = tf.concat([
        tf.reshape(tf.expand_dims(
            slot_gate_split[0], 2) * update_inputs_dim64_concat, (-1, num_dim64_slots * 64)),
        tf.reshape(tf.expand_dims(
            slot_gate_split[1], 2) * update_inputs_dim8_concat, (-1, num_dim8_slots * 8))
    ], 1)
    # inputs_slot_gated = norm(inputs_slot_gated, norm_type=variable_norm_type, scope_name="inputs_slot_gated")
    # query_input
    rown = tf.shape(inputs_slot_gated)[0]
    col = inputs_slot_gated.get_shape()[1]
    query_input = tf.reshape(inputs_slot_gated, (rown, 1, col))
    return query_input
def prepare_label_input(label_inputs, token_dim):
    labels = tf.concat(label_inputs, axis=-1)
    label_token = mio_dense_layer(labels, token_dim, None,  
                                  "label_token", "label_token_param",
                                  bias=False)
    label_token = tf.expand_dims(label_token, axis=1)
    return label_token
def prepare_user_profile_ctx(token_dim):
    # dense feas
    dense_feas = [user_embedding_64, user_embedding_8, user_user_lhuc, user_mcp_user, 
                  user_evtr_ids, user_clevtr_ids, user_peak_evtr_ids, user_ptr_ids, 
                  user_wtr_ids, user_epstr_ids, user_ltr_ids, user_ftr_ids, user_dtr_ids,
                  user_lvtr_ids, user_svr_ids, user_live_ids, user_cpr_ids, user_dfvtr_ids, 
                  user_thanos_etcm_ids, user_cmtr_ids, user_cmef_ids, user_hot_lvtr_ids, 
                  user_profile_time_cnt_id, user_watch_time_ids, user_watch_time_ids2, user_mmu_ids, 
                  user_fountain_evtr_v2_ids, user_ctr_input_64, user_ctr_input_8, user_ctr_ids, 
                  user_ctr_lhuc, user_ctr_lhuc_v1, user_fountain_enter_ids, user_fountain_effective_consumer_ids, 
                  user_wtd_v2_bias, user_wtd_v2_ids, user_adp_wtd_bias, user_adp_wtd_ids, user_mix_tabinfo, 
                  user_mix_ids, user_wtd_evtr_tabinfo, user_wtd_lvtr_tabinfo, user_pro_ids, user_pro_tabinfo, 
                  user_l2r_ids, user_l2r_tabinfo, user_local_life_city_id, user_base_tab_id, 
                  user_base_profile_feature]
    
    token_list = []
    for idx, fea in enumerate(dense_feas):
        token_list.append(
            tf.expand_dims(mio_dense_layer(fea, token_dim, None, 
                            f"dense_token_proj_fea_{idx}", 
                            f"dense_token_proj_fea_{idx}_param", bias=False), 1)
        )
    
    # seq feas
    b = tf.shape(user_embedding_64)[0]
    short_term_pids = tf.reshape(user_short_term_pids, (b, short_term_list_seq_len, 64))
    short_term_aids = tf.reshape(user_short_term_aids, (b, short_term_list_seq_len, 64))
    short_term_tags = tf.reshape(user_short_term_tags, (b, short_term_list_seq_len, 8))
    short_term_times = tf.reshape(user_short_term_times, (b, short_term_list_seq_len, 8))
    short_term_play = tf.reshape(user_short_term_play, (b, short_term_list_seq_len, 8))
    short_term_list_input = tf.concat([short_term_pids, short_term_aids, 
                                       short_term_tags, short_term_times, 
                                       short_term_play], 2)
    token_list.append(
        mio_dense_layer(short_term_list_input, token_dim, None, 
                        f"dense_token_proj_short_term_list_input", 
                        f"dense_token_proj_short_term_list_input_param", bias=False)
    )
    return tf.concat(token_list, 1)
def prepare_user_long_term_history(d_model):
    # if args.mode == "train":
    #     # compress by user_id_or_device_id_index
    #     with tf.xla.experimental.jit_scope(compile_ops=False):
    #         user_batch_size = tf.shape(item_long_term_pids3)[0]
    #         long_term_pids3 = tf.reshape(item_long_term_pids3, (user_batch_size, long_term_seq_len, 64))
    #         long_term_aids3 = tf.reshape(item_long_term_aids3, (user_batch_size, long_term_seq_len, 64))
    #         long_term_label3 = tf.reshape(item_long_term_label3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_tag3 = tf.reshape(item_long_term_tag3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_duration3 = tf.reshape(item_long_term_duration3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_play_time3 = tf.reshape(item_long_term_play_time3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_channel3 = tf.reshape(item_long_term_channel3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_play_x_duration3 = tf.reshape(item_long_term_play_x_duration3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_day_diff3 = tf.reshape(item_long_term_day_diff3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_mindiff3 = tf.reshape(item_long_term_mindiff3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_abspose3 = tf.reshape(item_long_term_abspose3, (user_batch_size, long_term_seq_len, 8))
    #         long_term_mask = tf.cast(tf.greater(colossus_time_s, 0), tf.int32) # [user_batch_size, long_term_seq_len]
    #         user_long_term_history = tf.concat([long_term_pids3, long_term_aids3, long_term_label3, 
    #                                         long_term_tag3, long_term_duration3, long_term_play_time3, 
    #                                         long_term_channel3, long_term_play_x_duration3, long_term_day_diff3, 
    #                                         long_term_mindiff3, long_term_abspose3], 2) # [user_batch_size, long_term_seq_len, 123]
    #         user_long_term_history = token_proj_tokens(user_long_term_history, d_model, "user_long_term_history") # [user_batch_size, long_term_seq_len, d_model]
    #         user_long_term_mask = tf.reshape(long_term_mask, (user_batch_size, long_term_seq_len))
    #         user_long_term_history = tf.gather(user_long_term_history, user_id_or_device_id_index) 
    #         user_long_term_mask = tf.gather(user_long_term_mask, user_id_or_device_id_index)
    # else:
    #     # expand, batch size
    #     batch_size = tf.shape(item_long_term_pids3)[0]
    #     long_term_pids3 = tf.reshape(item_long_term_pids3, (batch_size, long_term_seq_len, 64))
    #     long_term_aids3 = tf.reshape(item_long_term_aids3, (batch_size, long_term_seq_len, 64))
    #     long_term_label3 = tf.reshape(item_long_term_label3, (batch_size, long_term_seq_len, 8))
    #     long_term_tag3 = tf.reshape(item_long_term_tag3, (batch_size, long_term_seq_len, 8))
    #     long_term_duration3 = tf.reshape(item_long_term_duration3, (batch_size, long_term_seq_len, 8))
    #     long_term_play_time3 = tf.reshape(item_long_term_play_time3, (batch_size, long_term_seq_len, 8))
    #     long_term_channel3 = tf.reshape(item_long_term_channel3, (batch_size, long_term_seq_len, 8))
    #     long_term_play_x_duration3 = tf.reshape(item_long_term_play_x_duration3, (batch_size, long_term_seq_len, 8))
    #     long_term_day_diff3 = tf.reshape(item_long_term_day_diff3, (batch_size, long_term_seq_len, 8))
    #     long_term_mindiff3 = tf.reshape(item_long_term_mindiff3, (batch_size, long_term_seq_len, 8))
    #     long_term_abspose3 = tf.reshape(item_long_term_abspose3, (batch_size, long_term_seq_len, 8))
    #     long_term_mask = tf.cast(tf.greater(colossus_time_s, 0), tf.int32) # [batch_size, long_term_seq_len]
    #     user_long_term_history = tf.concat([long_term_pids3, long_term_aids3, long_term_label3, 
    #                                         long_term_tag3, long_term_duration3, long_term_play_time3, 
    #                                         long_term_channel3, long_term_play_x_duration3, long_term_day_diff3, 
    #                                         long_term_mindiff3, long_term_abspose3], 2) # [batch_size, long_term_seq_len, 123]
    #     user_long_term_history = token_proj_tokens(user_long_term_history, d_model, "user_long_term_history") # [batch_size, long_term_seq_len, d_model]
    #     user_long_term_mask = tf.reshape(long_term_mask, (batch_size, long_term_seq_len))
    # expand, batch size
    batch_size = tf.shape(item_long_term_pids3)[0]
    long_term_pids3 = tf.reshape(item_long_term_pids3, (batch_size, long_term_seq_len, 64))
    long_term_aids3 = tf.reshape(item_long_term_aids3, (batch_size, long_term_seq_len, 64))
    long_term_label3 = tf.reshape(item_long_term_label3, (batch_size, long_term_seq_len, 8))
    long_term_tag3 = tf.reshape(item_long_term_tag3, (batch_size, long_term_seq_len, 8))
    long_term_duration3 = tf.reshape(item_long_term_duration3, (batch_size, long_term_seq_len, 8))
    long_term_play_time3 = tf.reshape(item_long_term_play_time3, (batch_size, long_term_seq_len, 8))
    long_term_channel3 = tf.reshape(item_long_term_channel3, (batch_size, long_term_seq_len, 8))
    long_term_play_x_duration3 = tf.reshape(item_long_term_play_x_duration3, (batch_size, long_term_seq_len, 8))
    long_term_day_diff3 = tf.reshape(item_long_term_day_diff3, (batch_size, long_term_seq_len, 8))
    long_term_mindiff3 = tf.reshape(item_long_term_mindiff3, (batch_size, long_term_seq_len, 8))
    long_term_abspose3 = tf.reshape(item_long_term_abspose3, (batch_size, long_term_seq_len, 8))
    long_term_mask = tf.cast(tf.greater(colossus_time_s, 0), tf.int32) # [batch_size, long_term_seq_len]
    # train_step = config.get_step()
    # print_debug = conditional_tf_print(
    #     lambda: tf.equal(tf.mod(train_step, 10), 1),
    #     "step: ", train_step,
    #     "long_term_mask:", long_term_mask,
    #     output_stream=sys.stdout,
    #     summarize=20,
    # )
    # with tf.control_dependencies([print_debug]):
    #     long_term_mask = tf.identity(long_term_mask)
    # long_term_mask = P(long_term_mask, "long_term_mask", enable_print_in_training=True)
    user_long_term_history = tf.concat([long_term_pids3, long_term_aids3, long_term_label3, 
                                        long_term_tag3, long_term_duration3, long_term_play_time3, 
                                        long_term_channel3, long_term_play_x_duration3, long_term_day_diff3, 
                                        long_term_mindiff3, long_term_abspose3], 2) # [batch_size, long_term_seq_len, 123]
    user_long_term_history = token_proj_tokens(user_long_term_history, d_model, "user_long_term_history") # [batch_size, long_term_seq_len, d_model]
    user_long_term_mask = tf.reshape(long_term_mask, (batch_size, long_term_seq_len))
    return user_long_term_history, user_long_term_mask
def prepare_user_gsu_ctx():
    # TODO: add user gsu ctx
    user_gsu_ctx = tf.zeros((tf.shape(semantic_id_v2)[0], 1, 1), dtype=tf.float32) # mock user gsu ctx
    return user_gsu_ctx
def prepare_item_gsu_ctx(d_model):
    rag_list = [rag_item_local_life_emb, rag_item_embedding_64, 
                rag_item_embedding_8, rag_item_nebula_stats, rag_node_embedding]
    rag_items = tf.concat(rag_list, axis=-1) # [b, item_rag_size, dim]
    rag_items = rag_items[:, :rag_seq_len * token_rag_size, :]
    rag_items = mio_dense_layer(rag_items, d_model, None, "rag_items", "rag_items_param", bias=False) # [b, rag_seq_len, d_model]
    return rag_items
########### feature prepare end ###########
########### llama layer utils ###########
def rms_norm(x, scope_name, eps=1e-6):
    scale = tf.get_variable(f'scale_{scope_name}', x.shape.as_list()[-1], initializer=tf.ones_initializer())
    mean_square = tf.reduce_mean(x ** 2, -1, keep_dims=True)
    return scale * x * tf.rsqrt(mean_square + eps)
def ffn_layer(x, ffn_dim, hidden_dim, name):
    g = mio_dense_layer(x, ffn_dim, None, 
                        f"{name}_ffn_gate", 
                        f"{name}_ffn_gate_param", bias=False)
    
    v = mio_dense_layer(x, ffn_dim, None, 
                        f"{name}_ffn_value", 
                        f"{name}_ffn_value_param", bias=False)
    
    x = swish(g) * v
    x = mio_dense_layer(x, hidden_dim, None, 
                        f"{name}_ffn_output", 
                        f"{name}_ffn_output_param", bias=False)
    return x
def attention_layer(query, key, value, num_heads, att_emb_size, name, kv_mask=None):
    # query: [b, sq, emb]
    # key: [b, sk, emb]
    # value: [b, sk, emb]
    # num_heads: int
    # att_emb_size: int
    # kv_mask: [b, sk] 1 for valid, 0 for invalid
    batch_size = tf.shape(query)[0]
    seq_len_q = query.get_shape()[1]
    seq_len_k = key.get_shape()[1]
    print(f"seq_len_q: {seq_len_q}, seq_len_k: {seq_len_k}")
    hidden_dim = int(num_heads * att_emb_size)
    q = mio_dense_layer(query, hidden_dim, None, f"{name}_q_weight", f"{name}_q_weight_param", bias=False)
    k = mio_dense_layer(key, hidden_dim, None, f"{name}_k_weight", f"{name}_k_weight_param", bias=False)
    v = mio_dense_layer(value, hidden_dim, None, f"{name}_v_weight", f"{name}_v_weight_param", bias=False)
    q = tf.reshape(q, [batch_size, seq_len_q, num_heads, att_emb_size])
    k = tf.reshape(k, [batch_size, seq_len_k, num_heads, att_emb_size])
    v = tf.reshape(v, [batch_size, seq_len_k, num_heads, att_emb_size])
    q = tf.transpose(q, [0, 2, 1, 3]) # [b, h, sq, head_dim]
    k = tf.transpose(k, [0, 2, 1, 3]) # [b, h, sk, head_dim]
    v = tf.transpose(v, [0, 2, 1, 3]) # [b, h, sk, head_dim]
    attn_scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(att_emb_size) # [b, h, sq, sq]
    if kv_mask is not None:
        kv_mask = 1 - tf.reshape(kv_mask, [1, 1, seq_len_q, seq_len_k]) # [1,1,s,s]
        kv_mask = tf.tile(kv_mask, [batch_size, num_heads, 1, 1]) * -1e9 # [b, h, s, s]
        attn_scores = attn_scores + kv_mask
    attn_weights = tf.nn.softmax(attn_scores, axis=-1)
    attn_output = tf.matmul(attn_weights, v) # [b, h, sq, head_dim]
    
    attn_output = tf.transpose(attn_output, [0, 2, 1, 3]) # [b, sq, h, head_dim]
    attn_output = tf.reshape(attn_output, [batch_size, seq_len_q, num_heads * att_emb_size])
    attn_output = mio_dense_layer(attn_output, query.get_shape()[-1], None, 
                                  f"{name}_attn_output", 
                                  f"{name}_attn_output_param", bias=False)
    return attn_output
def build_flash_input(inputs, mask=None):
    # inputs: [b, sq, emb]
    # mask: [b, sq] 1 for valid, 0 for invalid
    batch_size = tf.shape(inputs)[0]
    seq_len_q = inputs.get_shape()[1]
    hidden_dim = inputs.get_shape()[2]
    if mask is None:
        mask = tf.ones((batch_size, seq_len_q), dtype=tf.int32)
    
    flat_inputs = tf.reshape(inputs, [batch_size * seq_len_q, hidden_dim]) # [b * sq, emb]
    flat_mask = tf.reshape(mask, [batch_size * seq_len_q]) # [b * sq]
    flat_inputs_with_mask = tf.boolean_mask(flat_inputs, flat_mask) # [valid_token_num, emb]
    batch_seq_len = tf.reduce_sum(mask, axis=1) # [b]
    batch_valid = batch_seq_len > 0
    seq_info = tf.boolean_mask(batch_seq_len, batch_valid) # [valid_seq_num] sum(seq_info) == flat_inputs_with_mask.shape[0]
    batch_valid_indices = tf.where(batch_valid)[:, 0]
    return flat_inputs_with_mask, seq_info, batch_valid_indices, batch_size
def restore_from_flash_input(flat_inputs, seq_info, batch_valid_indices, batch_size, seq_len, hidden_dim):
    # flat_inputs: [valid_token_num, hidden_dim]
    # seq_info: [valid_seq_num]
    # batch_valid_indices: [valid_batch_num]
    
    # flat buffer
    restored_inputs = tf.zeros([batch_size * seq_len, hidden_dim])
    seq_offsets = tf.cumsum(seq_info, exclusive=True) 
    batch_indices = tf.range(tf.shape(seq_info)[0])
    token_batch_indices = tf.repeat(batch_indices, seq_info)
    row_idx = tf.gather(batch_valid_indices, token_batch_indices)
    
    token_offsets = tf.range(tf.reduce_sum(seq_info))
    col_idx = token_offsets - tf.gather(seq_offsets, token_batch_indices)
    final_positions = tf.cast(row_idx, tf.int32) * seq_len + tf.cast(col_idx, tf.int32)
    final_positions = tf.expand_dims(final_positions, 1)
    
    restored_inputs = tf.scatter_nd(
        final_positions,
        flat_inputs,
        [batch_size * seq_len, hidden_dim]
    )
    restored_inputs = tf.reshape(restored_inputs, [batch_size, seq_len, hidden_dim])
    return restored_inputs
def attention_layer_flash(query, key, value, num_heads, 
                          att_emb_size, name, q_seqinfo, 
                          kv_seqinfo):
    # query:[1, valid_q_token_num, emb]
    # key:[1, valid_kv_token_num, emb]
    # value:[1, valid_kv_token_num, emb]
    # num_heads: int
    # att_emb_size: int
    # name: str
    # q_seqinfo: [seq_len]
    # kv_seqinfo: [seq_len]
    valid_q_token_num = tf.shape(query)[1]
    valid_kv_token_num = tf.shape(key)[1]
    hidden_dim = int(num_heads * att_emb_size)
    q_flat = mio_dense_layer(query, hidden_dim, None, f"{name}_q_weight", f"{name}_q_weight_param", bias=False)
    k_flat = mio_dense_layer(key, hidden_dim, None, f"{name}_k_weight", f"{name}_k_weight_param", bias=False)
    v_flat = mio_dense_layer(value, hidden_dim, None, f"{name}_v_weight", f"{name}_v_weight_param", bias=False)
    q_flat = tf.reshape(q_flat, [1, -1, num_heads, att_emb_size]) # [1, valid_q, h, head_dim]
    k_flat = tf.reshape(k_flat, [1, -1, num_heads, att_emb_size]) # [1, valid_k, h, head_dim]
    v_flat = tf.reshape(v_flat, [1, -1, num_heads, att_emb_size]) # [1, valid_v, h, head_dim]
    mask_type = config.nn.CustomMaskType.BlockDiagonalMask
    attn_out = config.mem_eff_attn(
        q_flat, k_flat, v_flat,
        bf16=True,
        q_seqinfo=q_seqinfo, k_seqinfo=kv_seqinfo,
        custom_mask_type=mask_type)
    
    attn_out = tf.reshape(attn_out, [valid_q_token_num, hidden_dim]) # [valid_q, emb]
    attn_out = mio_dense_layer(attn_out, hidden_dim, None, 
                              f"{name}_attn_output", 
                              f"{name}_attn_output_param", bias=False)
    
    return attn_out
def decoder_layer(x, ffn_dim, hidden_dim, name, num_heads, 
                  dropout_rate=0.0, 
                  use_flash_attention=False, 
                  align_score=False):
    # x: [b, sq, emb]
    assert hidden_dim % num_heads == 0
    att_emb_size = hidden_dim // num_heads
    res = x
    x = rms_norm(x, f"{name}_pre_norm")
    def flash_attn(x, num_heads, att_emb_size):
        batch_size = tf.shape(x)[0]
        seq_len_x = x.get_shape()[1]
        dim = x.get_shape()[2]
        flat_x = tf.reshape(x, [1, batch_size * seq_len_x, dim]) # [1, batch_size * seq_len_x, emb]
        seq_info = tf.ones([batch_size], dtype=tf.int32) * seq_len_x
        x = attention_layer_flash(flat_x, flat_x, flat_x, num_heads, 
                                  att_emb_size, f"{name}_attn", 
                                  seq_info, seq_info) # [batch_size * seq_len_x, emb]
        x = tf.reshape(x, [batch_size, seq_len_x, dim])
        return x
    
    def naive_attn(x, num_heads, att_emb_size):
        seq_len = tf.shape(x)[1]
        mask = tf.ones((seq_len, seq_len), dtype=tf.float32)
        mask = tf.linalg.band_part(mask, -1, 0) # 1 for valid, 0 for invalid
        x = attention_layer(x, x, x, num_heads, att_emb_size, f"{name}_attn", mask)
        return x
    
    assert not use_flash_attention and not align_score, "flash attention not support causal"
    if not align_score:
        if use_flash_attention:
            x = flash_attn(x, num_heads, att_emb_size)
        else:
            x = naive_attn(x, num_heads, att_emb_size)
    else:
        with tf.variable_scope(f"{name}_attn", reuse=tf.AUTO_REUSE):
            x1 = flash_attn(x, num_heads, att_emb_size)
            x2 = naive_attn(x, num_heads, att_emb_size)
            diff = tf.abs(x1 - x2)
            diff_ratio = diff / (tf.maximum(tf.abs(x1), tf.abs(x2)) + 1e-6) 
            train_step = config.get_step()
            print_debug = conditional_tf_print(
                lambda: tf.equal(tf.mod(train_step, 10), 1),
                f"{name} decoder_layer align_score",
                "step: ", train_step,
                "diff_max:", tf.reduce_max(diff),
                "diff_min:", tf.reduce_min(diff),
                "diff_mean:", tf.reduce_mean(diff),
                "diff_ratio_max:", tf.reduce_max(diff_ratio),
                "diff_ratio_min:", tf.reduce_min(diff_ratio),
                "diff_ratio_mean:", tf.reduce_mean(diff_ratio),
                output_stream=sys.stdout,
                summarize=20,
            )
            with tf.control_dependencies([print_debug]):
                x = tf.identity(x1)
    x = res + x
    # ffn
    res = x
    x = rms_norm(x, f"{name}_ffn_norm")
    x = ffn_layer(x, ffn_dim, hidden_dim, f"{name}_ffn")
    if dropout_rate > 0:
        x = tf.nn.dropout(x, rate=dropout_rate)
    x = x + res
    return x
def build_llama_model(inputs, ffn_dim, hidden_dim, name, num_heads, num_layers):
    # inputs: [b, sq, emb]
    # ffn_dim: int
    # hidden_dim: int
    # name: str
    # num_heads: int
    # num_layers: int
    # kv_mask: [b, sq] 
    # return: [b, sq, emb]
    batch_size = tf.shape(inputs)[0]
    seq_len = inputs.get_shape()[1]
    emb_size = inputs.get_shape()[2]
    # no need causal mask, use pos embedding instead
    # # causal mask
    # causal_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0) # [seq_len, seq_len]
    # causal_mask = causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 1) # [1, 1, seq_len, seq_len]
    
    pos_embedding = tf.get_variable(f"{name}_pos_embedding", (seq_len, emb_size))
    pos_embedding = tf.tile(tf.expand_dims(pos_embedding, 0), [batch_size, 1, 1])
    inputs = inputs + pos_embedding 
    for layer_idx in range(num_layers):
        inputs = decoder_layer(inputs, ffn_dim, hidden_dim, 
                               f"{name}_layer_{layer_idx}", 
                               num_heads, kv_mask=None)
        
    # final_ln
    inputs = rms_norm(inputs, f"{name}_final_ln")
    return inputs
def cross_former_layer(q, kv, ffn_dim, hidden_dim, 
                       name, num_heads, 
                       kv_mask=None, dropout_rate=0.0,
                       use_flash_attention=False,
                       align_score=False):
    # q: [b, sq, emb]
    # kv: [b, sk, emb]
    # ffn_dim: int
    # hidden_dim: int
    # name: str
    # num_heads: int
    # num_layers: int
    # return: [b, sq, emb]
    # qproj & kvproj
    assert hidden_dim % num_heads == 0
    att_emb_size = hidden_dim // num_heads
    res = q
    q = rms_norm(q, f"{name}_pre_norm")
    q = attention_layer(q, kv, kv, num_heads, 
                        att_emb_size, f"{name}_attn", kv_mask)
    q = res + q
    res = q
    q = rms_norm(q, f"{name}_ffn_norm")
    q = ffn_layer(q, ffn_dim, hidden_dim, f"{name}_ffn")
    if dropout_rate > 0:
        q = tf.nn.dropout(q, rate=dropout_rate)
    q = res + q
    return q
def build_cross_former_self_layers(q, kv, cross_ffn_dim, self_ffn_dim, hidden_dim, 
                              name, 
                              num_heads, num_layers, 
                              kv_mask=None, dropout_rate=0.0):
    batch_size = tf.shape(q)[0]
    seq_len = q.get_shape()[1]
    emb_size = q.get_shape()[2]
    pos_embedding = tf.get_variable(f"{name}_pos_embedding", (seq_len, emb_size))
    pos_embedding = tf.tile(tf.expand_dims(pos_embedding, 0), [batch_size, 1, 1])
    q = q + pos_embedding
    kv = rms_norm(kv, f"{name}_kv_norm")
    for layer_idx in range(num_layers):
        q = cross_former_layer(q, kv, cross_ffn_dim, hidden_dim, 
                               f"{name}_cross_layer_{layer_idx}", 
                               num_heads, None, dropout_rate,
                               use_flash_attention=use_flash_attention,
                               align_score=False)
        
        q = decoder_layer(q, self_ffn_dim, hidden_dim, 
                          f"{name}_self_layer_{layer_idx}", 
                          num_heads, dropout_rate, 
                          use_flash_attention=False, align_score=False)
        
    q = rms_norm(q, f"{name}_final_norm")
    return q
def build_cross_former_layers(q, kv, ffn_dim, hidden_dim, 
                              name, 
                              num_heads, num_layers, 
                              kv_mask=None, dropout_rate=0.0):
    # kv pre norm
    kv = rms_norm(kv, f"{name}_kv_norm")
    for layer_idx in range(num_layers):
        q = cross_former_layer(q, kv, ffn_dim, hidden_dim, f"{name}_layer_{layer_idx}", num_heads, kv_mask, dropout_rate)
    return q
def build_query_former_layers(inputs, query_num, ffn_dim, hidden_dim, 
                              name, 
                              num_heads, num_layers, 
                              kv_mask=None, dropout_rate=0.0):
    batch_size = tf.shape(inputs)[0]
    # build_query
    query_embs = tf.get_variable(f"{name}_query_embs", (query_num, hidden_dim))
    query_embs = tf.tile(tf.expand_dims(query_embs, 0), [batch_size, 1, 1])
    query_embs = build_cross_former_layers(query_embs, inputs, ffn_dim, hidden_dim, 
                                           name, 
                                           num_heads, num_layers, 
                                           kv_mask, dropout_rate)
    return query_embs
        
def fr_model():
    # global config
    d_model = 640
    num_layers = 8
    ffw_size = int(d_model * 4)
    num_heads = 10
    
    vocab_size = 64
    item_seq_len = senmantic_id_token_num
    # field pre process
    hidden_dim = d_model
    cross_ffn_dim = int(4 * hidden_dim)
    with tf_name_scope("pre_process"), new_xla_jit_context():
        item_token_label = semantic_id_v2
        item_token_embedding, all_zeros_mask = prepare_item_tokens(vocab_size, item_seq_len, d_model)
        # label_input = prepare_label_input(labels, d_model)
        user_profile_ctx = prepare_user_profile_ctx(token_dim=d_model)
        user_long_term_history, user_long_term_mask = prepare_user_long_term_history(d_model) # [user_batch_size, ***]
        context = tf.concat([user_profile_ctx, user_long_term_history], axis=1)
        # batch_size = tf.shape(user_long_term_mask)[0]
        # ctx_seq_len = context.get_shape()[1]
        # ctx_kv_mask = tf.concat([tf.ones([batch_size, ctx_seq_len - user_long_term_mask.get_shape()[1]], dtype=tf.int32), user_long_term_mask], axis=1)
    
    with tf_name_scope("qformer"), new_xla_jit_context():
        bos_token_emb = tf.get_variable(f"bos_token_emb", (1, 1, d_model))
        bos_token_emb = tf.tile(bos_token_emb, [tf.shape(item_token_embedding)[0], 1, 1])
        item_token_embedding = tf.concat([bos_token_emb, item_token_embedding[:, :-1, :]], axis=1) # [b, item_seq_len, d_model]
        qformer_res = build_cross_former_self_layers(item_token_embedding, 
                                                     context, 
                                                     cross_ffn_dim, ffw_size, hidden_dim, 
                                                     "qformer", 
                                                     num_heads, num_layers, 
                                                     kv_mask=None, dropout_rate=0.0)
    with tf_name_scope("item_pred_head"), new_xla_jit_context():
        b = tf.shape(qformer_res)[0]
        item_pred_head = mio_dense_layer(qformer_res, vocab_size, 
                                         None, "item_pred_head", "item_pred_head_param", bias=False)
        item_pred_logits = tf.reshape(item_pred_head, (b, item_seq_len, vocab_size))  # logits already shifted
    
    return (
        item_pred_logits,  # [b, item_seq_len, vocab_size - 1]
        item_token_label,  # [b, item_seq_len]
        all_zeros_mask     # [b]
    )
def select_tokens(sequence, indices):
    batch_size = tf.shape(sequence)[0]
    batch_indices = tf.range(batch_size, dtype=indices.dtype)
    batch_indices = tf.expand_dims(batch_indices, 1)
    
    # batchid, tokenid
    gather_indices = tf.concat([batch_indices, indices], axis=1)
    selected = tf.gather_nd(sequence, gather_indices)
    return selected
if args.mode == "train":
    eval_targets = []
    tab_bits = tf.cast(config.get_dense_fea("tab_bits", 1, dtype=tf.int64), tf.int32)
    ones = tf.fill(tf.shape(tab_bits), 1.0)
    zeros = tf.fill(tf.shape(tab_bits), 0.0)
    token_num = senmantic_id_token_num
    item_pred_logits, item_token_label, all_zeros_mask = fr_model()
    token_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=item_token_label,
        logits=item_pred_logits
    ) # [b, item_seq_len]
    all_zeros_mask = tf.cast(all_zeros_mask, tf.float32) # [b] 1 for all zeros
    all_zeros_mask_expand = tf.tile(tf.expand_dims(all_zeros_mask, 1), [1, token_num]) # [b, item_seq_len]
    item_token_mask_float = 1.0 - all_zeros_mask_expand # [b, item_seq_len]
    masked_token_loss = token_loss * item_token_mask_float # [b, item_seq_len]
    loss = tf.reduce_sum(masked_token_loss) / (tf.reduce_sum(item_token_mask_float) + 1e-6)
    total_loss = loss * tf.cast(tf.shape(item_token_label)[0], tf.float32)

    eval_targets.append(("loss", loss * ones, zeros, ones, "linear_regression"))
    tf.summary.scalar("global_metric/loss", loss)
    # NOTE: just for trigger model dump
    eval_targets.append(("adp_evtr", loss * ones, zeros, ones, "linear_regression")) 
    pred_token = tf.argmax(item_pred_logits, axis=2, output_type=tf.int32) # [b, item_seq_len]
    target_token = tf.cast(item_token_label, tf.int32) # [b, item_seq_len]
    rebuild_token = pred_token
    # loss by token
    for i in range(token_num):
        pos_loss = masked_token_loss[:, i]
        pos_mask = item_token_mask_float[:, i]
        pos_loss = tf.reduce_sum(pos_loss) / (tf.reduce_sum(pos_mask) + 1e-6)
        eval_targets.append(("pos_{}_avg_loss".format(i), pos_loss * ones, zeros, ones, "linear_regression"))
        tf.summary.scalar("pos_loss/pos_{}_avg_loss".format(i), pos_loss)
        pos_target = item_token_label[:, i]
        pos_pred = pred_token[:, i]
        pos_equal = tf.cast(tf.equal(pos_pred, pos_target), tf.float32) * pos_mask
        pos_acc = tf.reduce_sum(pos_equal) / (tf.reduce_sum(pos_mask) + 1e-6)
        eval_targets.append(("pos_{}_acc".format(i), pos_acc * ones, zeros, ones, "linear_regression"))
        tf.summary.scalar("pos_acc/pos_{}_acc".format(i), pos_acc)
    # acc metric
    equal_token = tf.cast(tf.equal(pred_token, target_token), tf.float32) * item_token_mask_float
    acc = tf.reduce_sum(equal_token) / (tf.reduce_sum(item_token_mask_float) + 1e-6)
    eval_targets.append(("acc", acc * ones, zeros, ones, "linear_regression"))
    tf.summary.scalar("global_metric/acc", acc)
    # item rebuild hit rate
    valid_sample = 1.0 - tf.cast(all_zeros_mask, tf.float32) # [b]
    valid_sample_ratio = tf.reduce_sum(valid_sample) / (tf.reduce_sum(tf.ones_like(valid_sample)) + 1e-6)
    eval_targets.append(("valid_sample_ratio", valid_sample_ratio * ones, zeros, ones, "linear_regression"))
    tf.summary.scalar("global_metric/valid_sample_ratio", valid_sample_ratio)
    item_token_hit_num = tf.reduce_sum(tf.cast(tf.equal(rebuild_token, target_token), tf.float32), axis=1) # [b]
    hit_token = tf.cast(tf.equal(item_token_hit_num, token_num), tf.float32) * valid_sample # [b]
    hit_rate = tf.reduce_sum(hit_token) / (tf.reduce_sum(valid_sample) + 1e-6)
    eval_targets.append(("item_hit_rate", hit_rate * ones, zeros, ones, "linear_regression"))
    tf.summary.scalar("global_metric/item_hit_rate", hit_rate)
    for k, lb in labels_map.items():
        lb = tf.cast(lb, tf.float32)
        lb_avg = tf.reduce_mean(lb)
        eval_targets.append((f"lb_avg_{k}", lb_avg * ones, zeros, ones, "linear_regression"))
        tf.summary.scalar(f"lb_ratio/lb_avg_{k}", lb_avg)
    if args.with_kai_v2 and not args.local_debug:
        # config.set_slot_param_attr([344, 348], config.nn.ParamAttr(access_method=config.nn.ProbabilityAccess(100.0),
        #                                                            recycle_method=config.nn.UnseendaysRecycle(30, 2.0)))
        sparse_optimizer = config.optimizer.Adam(0.0005) # freeze embedding
        dense_optimizer_bias = config.optimizer.Adam(0.001)
        dense_optimizer_mlp = config.optimizer.AdamW(learning_rate=0.001, weight_decay=0.001)
        dense_var_bias_list = []
        dense_var_mlp_list = []
        for var in config.get_dense_trainable_variables():
            if "bias" in var.name or "norm" in var.name:
                dense_var_bias_list.append(var)
            else:
                dense_var_mlp_list.append(var)
        print("dense_var_bias_list:", dense_var_bias_list)
        sparse_optimizer.minimize(total_loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
        dense_optimizer_bias.minimize(total_loss, var_list=dense_var_bias_list)
        dense_optimizer_mlp.minimize(total_loss, var_list=dense_var_mlp_list)
    elif args.local_debug:
        # 本地调试模式下使用TensorFlow的优化器
        print("[本地调试] 使用TensorFlow优化器")
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(total_loss)
        
        # 在本地调试模式下，使用TensorFlow会话运行模型
        print("[本地调试] 初始化TensorFlow会话")
        init_op = tf.global_variables_initializer()
        
        # 创建TensorFlow会话并运行模型
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True  # 动态分配GPU内存
        
        with tf.Session(config=sess_config) as sess:
            print("[本地调试] 运行变量初始化")
            sess.run(init_op)
            
            print("[本地调试] 运行一次前向计算")
            loss_val = sess.run(loss)
            print(f"[本地调试] 损失值: {loss_val}")
            
            print("[本地调试] 运行一次反向传播和参数更新")
            _, loss_val = sess.run([train_op, loss])
            print(f"[本地调试] 更新后的损失值: {loss_val}")
            
            # 打印训练张量的形状
            print("\n[本地调试] 查看模型中的一些关键张量形状:")
            for name, tensor in mock_tensors.items():
                if random.random() < 0.2:  # 随机打印部分张量信息
                    tensor_val = sess.run(tensor)
                    print(f"  {name}: 形状={tensor_val.shape}, 均值={np.mean(tensor_val):.4f}, 标准差={np.std(tensor_val):.4f}")
    else:
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        opt = optimizer.minimize(loss)
    if args.dryrun and not args.local_debug:
        config.mock_and_profile(loss, "./training_log/", batch_sizes=[8192])
    elif args.with_kai_v2 and not args.local_debug:
        config.build_model(
            optimizer=[sparse_optimizer, dense_optimizer_bias, dense_optimizer_mlp], metrics=eval_targets)
    elif args.local_debug:
        # 本地调试模式已经在前面运行了模型
        pass
    else:
        config.dump_training_config(
            "./training/conf", eval_targets, opts=[opt], text=args.text)
else:
    # item_pred_logits: [b, item_seq_len, vocab_size - 1]
    # item_token_mask: [b, item_seq_len]
    # item_token_label: [b, item_seq_len]
    # return_logits = True
    # return_type = "vallina_topk" # ["logits", "vallina_topk", "gumble_topk"]
    return_type = "beam"
    token_num = senmantic_id_token_num
    item_pred_logits, item_token_label, _ = fr_model()
    item_pred_logits = tf.cast(item_pred_logits, tf.float32)
    top_k = 50
    batch_size = tf.shape(item_pred_logits)[0]
    pred_token_num = item_pred_logits.shape[1]
    vocab_size = item_pred_logits.shape[2]

    print(f"batch_size: {batch_size}, pred_token_num: {pred_token_num}, vocab_size: {vocab_size}")
    
    if return_type == "logits":
        item_logits = tf.reshape(item_pred_logits, (batch_size, pred_token_num * vocab_size))
        targets = [("logits", item_logits)]
    elif return_type == "beam":
        cur_step = config.get_dense_fea("cur_step", 1, dtype=tf.int32)
        cur_step = tf.reshape(tf.cast(cur_step, tf.int32), (batch_size, 1))
        batch_mask = config.get_dense_fea("batch_mask", 1, dtype=tf.float32)
        batch_mask = (1 - tf.reshape(batch_mask, (batch_size, 1))) * (-1e9)
        batch_mask = tf.tile(batch_mask, [1, vocab_size])
        batch_mask = tf.reshape(batch_mask, (1, batch_size * vocab_size))
        token_logits = select_tokens(item_pred_logits, cur_step)
        token_logits = tf.reshape(token_logits, (batch_size, vocab_size))
        p_temp = config.get_dense_fea("p_temp", 1, dtype=tf.float32)
        p_temp = tf.cast(p_temp[0][0], tf.float32)
        
        token_prob = tf.nn.softmax(token_logits / p_temp, axis=-1) # [b, vocab_size]
        token_logprob = tf.nn.log_softmax(token_logits / p_temp, axis=-1) # [b, vocab_size]
        beam_prob = config.get_dense_fea("beam_prob", 1, dtype=tf.float32) # [b, 1]
        beam_probs = beam_prob + token_logprob # [b, vocab_size]
        beam_probs = tf.reshape(beam_probs, (1, batch_size * vocab_size))
        beam_probs = beam_probs + batch_mask
        topk_beam_prob, topk_indices = tf.nn.top_k(beam_probs, k=batch_size)
        topk_prob = tf.reshape(tf.gather(tf.reshape(token_prob, (1, batch_size * vocab_size)), topk_indices, batch_dims=1), (batch_size, 1))
        topk_beam_prob = tf.reshape(topk_beam_prob, (batch_size, 1))
        topk_indices = tf.reshape(topk_indices, (batch_size, 1))
        targets = [
            ("topk_prob", topk_prob), 
            ("topk_indices", topk_indices),
        ]
    elif return_type == "vallina_topk":
        p_temp = config.get_dense_fea("p_temp", 1, dtype=tf.float32)
        p_temp = tf.cast(p_temp[0][0], tf.float32)
        item_pred_prob = tf.nn.softmax(item_pred_logits / p_temp, axis=-1)
        topk_prob, topk_indices = tf.nn.top_k(item_pred_prob, k=top_k)
        topk_prob = tf.reshape(topk_prob, (batch_size, pred_token_num * top_k))
        topk_indices = tf.reshape(topk_indices, (batch_size, pred_token_num * top_k))
        targets = [
            ("topk_prob", topk_prob),
            ("topk_indices", topk_indices),
        ]
    elif return_type == "gumble_topk":
        temperature = 0.0
        # add gumbel noise
        if temperature > 0.0:
            noise = tf.random.uniform(shape=tf.shape(item_pred_logits), dtype=tf.float32)
            gumbel_noise = (- tf.log(noise)) ** temperature
            item_pred_logits = tf.math.exp(item_pred_logits) / gumbel_noise
        
        prob = tf.nn.softmax(item_pred_logits, axis=-1) # [b, item_seq_len, vocab_size - 1]
        topk_prob, topk_indices = tf.nn.top_k(prob, k=top_k)  # [b, item_seq_len, top_k]
        batch_size = tf.shape(topk_prob)[0]
        topk_prob = tf.reshape(topk_prob, (batch_size, top_k * token_num ))
        topk_indices = tf.reshape(topk_indices, (batch_size, top_k * token_num))
        targets = [("topk_prob", topk_prob), ("topk_indices", topk_indices)]
    else:
        raise ValueError(f"not supported return type {return_type}")
    
    q_names, preds = zip(*targets)
    if args.dryrun and not args.local_debug:
        config.mock_and_profile(
            preds, "./predict_log/", batch_sizes=[200], compressed_embedding_size={"USER": 4})
    elif not args.local_debug:
        config.dump_predict_config(
            "./predict/causal_large_config", targets, input_type=3, extra_preds=q_names, text=args.text)
    else:
        init_op = tf.global_variables_initializer()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            res = sess.run(preds)
            for name, pred in zip(q_names, res):
                print(f"{name} shape: {pred.shape}")