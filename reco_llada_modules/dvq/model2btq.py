from typing import Optional, Dict, Tuple, List
import yaml
import logging
from collections import namedtuple
from dataclasses import dataclass
import hashlib

import math
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial

import struct
import numpy as np
from argparse import ArgumentParser
import time

import traceback
from framework_wrappers import BtqClient
from framework_wrappers import BtqException
from framework_wrappers import BtqErrorCode
from vqvae import VQVAE

# import kai_torch as KT

# from utils.serving_utils.proto.model_pb2 import ModelUpdateMessage, ModelItem

# 全局配置类
class GlobalConfig:
    def __init__(self, dnn_config_file):
        # 读取 YAML 文件
        with open(dnn_config_file, 'r') as file:
            config = yaml.safe_load(file)

        assert len(config['param']) > 0, f"Invalid param in dnn_yaml config: {dnn_config_file}!!!"
        self.dnn_params = config['param']

        total_param_num = sum([x['rown'] * x['coln'] for x in config['param']])
        print(f"Total param num:  {total_param_num / 1000**3} B")


# refer: teams/reco-model/base/mio/mio/include/mio/DnnOutput.h
# struct DnnOutput {
#   std::vector<float> dnn_weights;
#   std::vector<double> input_counts;
#   std::vector<double> input_sums;
#   std::vector<double> input_scales;
#   uint64_t version;

#   MIO_DEFINE_SIMPLE_SERIALIZER(DnnOutput, dnn_weights, input_counts, input_sums, input_scales, version)
# };
class DnnOutput:
    def __init__(self):
        self.dnn_weights = np.array([], dtype=np.float32)
        self.input_counts = np.array([], dtype=np.float64)
        self.input_sums = np.array([], dtype=np.float64)
        self.input_scales = np.array([], dtype=np.float64)
        self.version = 0

    def resize(self, size):
        self.dnn_weights = np.array([0.0] * size, dtype=np.float32)
    
    def clip_dnn_weights(self, clip_value):
        self.dnn_weights = np.clip(self.dnn_weights, -clip_value, clip_value)

    def serialize_weights(self):
        self.buffer = bytearray()
        # Serialize version
        self.buffer.extend(struct.pack('<Q', self.version))
        # Serialize size 
        self.buffer.extend(struct.pack('<Q', len(self.dnn_weights))) 
        for weight in self.dnn_weights:
            self.buffer.extend(struct.pack('<f', weight))
        return bytes(self.buffer)

    def serialize(self,):
        # TODO(huangrui06): 使用 BytesIO 提升性能
        self.buffer = bytearray()
        self.buffer.extend(struct.pack('<Q', len(self.dnn_weights))) 
        for weight in self.dnn_weights:
            self.buffer.extend(struct.pack('<f', weight))
        # Serialize input_counts
        self.buffer.extend(struct.pack('<Q', len(self.input_counts)))
        for count in self.input_counts:
            self.buffer.extend(struct.pack('<d', count))
        # Serialize input_sums
        self.buffer.extend(struct.pack('<Q', len(self.input_sums)))
        for sum_ in self.input_sums:
            self.buffer.extend(struct.pack('<d', sum_))

        # Serialize input_scales
        self.buffer.extend(struct.pack('<Q', len(self.input_scales)))
        for scale in self.input_scales:
            self.buffer.extend(struct.pack('<d', scale))

        # Serialize version
        self.buffer.extend(struct.pack('<Q', self.version))
        return bytes(self.buffer)

    @staticmethod
    def deserialize(data):
        pass

# /* 一个 Frame 是一个 dnn model 在发送时拆分得到的多个分片中的一个，之后这个分片
#  * 按照特定格式组装并序列化成字节流通过 btq 发送出去；模型分片的序列化方式参见:
#  * teams/reco-model/base/mio/mio/include/mio/Archive.h; 这里反序列化不采用
#  * mio/Archive.h 的实现，是因为其序列化方式是明确而简单的, 可以脱离 mio/Archive.h
#  * 基础库的支持
#  *
#  * 分片格式
#  * struct Frame {
#  *   uint64_t version;
#  *   uint64_t num_frames;
#  *   uint64_t frame_index;
#  *   std::string sub_message;
#  * };
#  *
#  */


class Frame:
    def __init__(self):
        self.version = 0
        self.num_frames = 0
        self.frame_index = 0
        self.size = 0
        self.frame_data = None

    def build(self, version, num_frames, frame_index, size, frame_data):
        self.version = version
        self.num_frames = num_frames
        self.frame_index = frame_index
        self.size = size
        self.frame_data = frame_data

    def serialize(self):
        self.buffer = bytearray()
        self.buffer.extend(struct.pack('<Q', self.version)) 
        self.buffer.extend(struct.pack('<Q', self.num_frames))
        self.buffer.extend(struct.pack('<Q', self.frame_index))
        self.buffer.extend(struct.pack('<Q', self.size))
        # Serialize frame_data 
        self.buffer.extend(self.frame_data)
        return bytes(self.buffer)

    @staticmethod
    def deserialize(data):
        pass
        
def build_frame_task(chunk_data, version, num_frames, idx, chunk_size):
    """优化后的任务函数"""
    try:
        frame = Frame()
        frame.build(
            version=version,
            num_frames=num_frames,
            frame_index=idx,
            size=chunk_size,
            frame_data=chunk_data,
        )
        return frame.serialize()
    except Exception as e:
        logging.error(f"Frame {idx} build failed: {e}")
        return None

def send_btq_batch(btq_topic, buffers):
    """批量发送到消息队列"""
    if not isinstance(buffers, list):
        buffers = [buffers]
    valid_buffers = [b for b in buffers if b is not None]
    if len(valid_buffers) > 0:
        BtqClient.produce(btq_topic, valid_buffers)
    failed_count = len(buffers) - len(valid_buffers)
    if failed_count > 0:
        logging.warning(f"Failed frames: {failed_count}")
    print(f"Successfully sent {len(valid_buffers)} frames to BTQ topic: {btq_topic}")

def build_frames_async(msg, version, btq_topic, frame_size=4_000_000):
    """优化后的主函数"""
    total_length = len(msg)
    print(f"Start building frames, size: {total_length}")
    num_frames = math.ceil(total_length / frame_size)
    print(f"total_length: {total_length}, num_frames: {num_frames}, frame_size: {frame_size}")
    
    # 生成任务参数（精简版）
    tasks = [
        (
            msg[i:min(i+frame_size, total_length)],
            version,
            num_frames,
            idx,
            min(frame_size, total_length - i)
        )
        for idx, i in enumerate(range(0, total_length, frame_size))
    ]

    # 动态调整进程数
    num_workers = min(cpu_count(), len(tasks))
    chunksize = max(1, len(tasks) // (num_workers * 4))
    print(f"Chunksize set to {chunksize}, num_workers: {num_workers}")
    
    with Pool(processes=num_workers) as pool:
        # 批量提交任务
        result = pool.starmap_async(
            build_frame_task,
            tasks,
            chunksize=chunksize,
        )

        buffer = []
        send_bz = 32
        for rst in result.get():
            buffer.append(rst)
            if len(buffer) >= send_bz:
                send_btq_batch(btq_topic, buffer[:send_bz])
                buffer = buffer[send_bz:]

        if len(buffer) > 0:
            send_btq_batch(btq_topic, buffer)

    print("Frame building completed")

# 从完整的模型 checkpoint 构造分片
def build_frames(msg, version, frame_size=4_000_000):
    """按照 frame_size 将 msg 分片
    msg: 待分片的模型 ckpt 二进制 bytes
    version: 当前模型 ckpt 版本
    frame_size: 字节
    """
    rst = list()
    total_length = len(msg)
    print(f"start build frames, total_length: {total_length}")
    num_frames = math.ceil(total_length / frame_size)
    print(f"total_length: {total_length}, num_frames: {num_frames}, frame_size: {frame_size}")
    for idx, i in enumerate(range(0, total_length, frame_size)):
        frame = Frame()
        begin_idx = i
        end_idx = min(i+frame_size, total_length)
        cur_frame_size = end_idx - begin_idx
        frame.build(
            version=version, 
            num_frames=num_frames,
            frame_index=idx,
            size=cur_frame_size,
            frame_data=msg[begin_idx:end_idx]
        )
        buffer = frame.serialize()
        rst.append(buffer)
    print(f"bulid frame finish")
    return rst
    
 
# refer: teams/reco-model/base/mio/mio/include/mio/Archive.h
def clip(value, clip_value=None):
    if clip_value is not None:
        return max(min(value, clip_value), -clip_value)
    else:
        return value


def convert_embedding_dtype(x, dtype=np.float32):
    assert dtype in ["mio_int16", np.float32, np.float16], "Invalid dtype!!!"
    if dtype == "mio_int16":
        return np.round(np.clip(x,-10,10) * 3000).astype(np.int16)
    else:
        return x.astype(dtype)
        

def calculate_md5(msg):
    md5_hash = hashlib.md5()
    md5_hash.update(msg)
    print(f'md5sum: {md5_hash.hexdigest()}')


@dataclass
class ShardSparseTask:
    emb_name: str
    emb_dtype: np.float32
    emb_slot: int
    keys: np.ndarray
    values: np.ndarray
    batch_size: int
    btq_prefix: str
    expire_timet: int
    model_time_ms: int
    debug_mode: bool = False

@dataclass
class ModelSenderConfig:
    debug_mode: bool
    sparse: dict
    dense: dict

class ModelSender:
    def __init__(self, config: ModelSenderConfig):
        self.debug_mode = config.get("debug_mode", False)
        
        sparse_config = config.get('sparse', {})
        self.emb_btq_prefix = sparse_config.get('btq_prefix', None)
        self.emb_btq_shard = sparse_config.get('btq_shard_num', None)
        self.emb_batch_size = sparse_config.get('batch_size', None)
        self.emb_configs = sparse_config.get('embeddings', None)
        self.emb_expire_timet = sparse_config.get('expire_timet', None)
        self.emb_send_queue_size = sparse_config.get("send_queue_size", 4)
        self.emb_send_queue = mp.Queue(self.emb_send_queue_size)

        dense_config = config['dense']
        self.dnn_yaml_file = dense_config['dnn_yaml_file']
        self.dnn_weight_clip = dense_config.get('dnn_weight_clip', None)
        self.dnn_btq_prefix = dense_config['btq_prefix']
        self.dnn_btq_frame_size = dense_config['btq_frame_size']
        self.dnn_config = GlobalConfig(self.dnn_yaml_file)
        self.dnn_params = self.dnn_config.dnn_params
        self.dnn_send_queue_size = dense_config.get("send_queue_size", 1)
        self.dnn_send_queue = mp.Queue(self.dnn_send_queue_size)

        # print(f"start send process")
        # self.start_send_process()

    def start_send_process(self):

        self.send_emb_proc = mp.Process(
            target=self.async_send_emb_proc, 
            args=(self.emb_send_queue,))
        self.send_emb_proc.start()

        self.send_dnn_proc = mp.Process(
            target=self.async_send_dnn_proc, 
            args=(self.dnn_send_queue,))
        self.send_dnn_proc.start()
        
    
    def async_send_emb_proc(self, send_queue, **kwargs):
        while True:
            data = send_queue.get()
            if data is None:
                continue 
            try:
                ret = self.SendSparseEmbeddings(*data)
                print(f"emb send_size: {ret}")
            except Exception as e:
                print(f"[emb send_error]: {traceback.format_exc()}")

    def async_send_dnn_proc(self, send_queue, **kwargs):
        while True:
            data = send_queue.get()
            if data is None:
                continue 
            try:
                ret = self.SendDNNModel(*data)
                print(f"dnn send_size: {ret}")
            except Exception as e:
                print(f"[dnn send_error]: {traceback.format_exc()}")
 
    def send_dnn_params(self, model_state_dict, model_time_ms=-1, check_consistency=False):
        model_state_dict= {f"model.{k}": v.cpu().numpy() for k, v in model_state_dict.items()}
        send_model_state_dict = self._process_ckpt(model_state_dict)
        KT.training.btq_save_dense(send_model_state_dict, model_time_ms)

    def async_send_dnn_params(self, model_state_dict, model_time_ms=-1, debug=False):
        model_state_dict_cpu = {f"model.{k}": v.cpu() for k, v in model_state_dict.items()}
        self.dnn_send_queue.put((model_state_dict_cpu, model_time_ms, debug))

    def _process_ckpt(self, model_state_dict):
        transposed = {
            p['th_tensor_name']: np.transpose(model_state_dict[p['th_tensor_name']]) 
            for p in self.dnn_params if p['need_transpose']
        }
        print(f'process keys'.center(100, '-'))
        print(f'transposed keys: {transposed.keys()}')
        model_state_dict.update(transposed)

        return model_state_dict

        
    def SendDNNModel(self, model_state_dict, model_time_ms=-1, check_consistency=False):
        if model_time_ms <= 0:
            logging.error(f"invalid model_time_ms: {model_time_ms}")
            return 0

        # build DnnOutput
        dnn_output = DnnOutput()
        dnn_output.version = model_time_ms
        send_online_size = 0
        for param in self.dnn_params:
            tensor = model_state_dict.get(param['th_tensor_name'])
            # 检查 pytorch 模型参数是否存在
            if tensor is None:
                logging.error(f"fail to find dense tensor: {param['th_tensor_name']}")
                return 0

            # 检查参数维度是否一致
            if param['rown'] * param['coln'] != tensor.numel():
                logging.error(f"dense tensor size not match, tensor: {param['th_tensor_name']}, expected: ({param['rown']},{param['coln']}), received: {tensor.shape}")

            send_online_size += tensor.numel()

        dnn_output.resize(send_online_size)

        # 填充模型参数
        dst_oft = 0
        for param in self.dnn_params:
            tensor = model_state_dict[param['th_tensor_name']].numpy()
            # 检查参数是否需要 transpose
            if param['need_transpose']:
                tensor = np.transpose(tensor)
            # 参数 flatten 
            src = tensor.flatten()
            len_ = tensor.size 
            dnn_output.dnn_weights[dst_oft:dst_oft+len_] = np.copy(src)
            dst_oft += len_
        
        if self.dnn_weight_clip is not None:
            dnn_output.clip_dnn_weights(self.dnn_weight_clip)

        if dst_oft != send_online_size:
            logging.error("dst_oft does not match send_online_size")
            return 0

        if check_consistency:
            msg = dnn_output.serialize_weights() # bytes
            # 校验一致性
            with open("./dnn_model_bin", "wb") as f:
                f.write(msg)
            calculate_md5(msg)
            print(f"version: {model_time_ms}")

        msg = dnn_output.serialize() # bytes
        if self.debug_mode:
            print("debug send_dense".center(100, '-'))
            print(f"dnn_output size: {send_online_size}")

        btq_topic = f"{self.dnn_btq_prefix}_dnn_model"

        BtqClient.produce(btq_topic, [msg])
        return len(msg)

        if True:
            frames = build_frames(msg, dnn_output.version, frame_size=self.dnn_btq_frame_size) # list of frames
            # write to btq
            for frame in frames:
                BtqClient.produce(btq_topic, [frame])
        else:
            build_frames_async(msg, dnn_output.version, btq_topic, frame_size=self.dnn_btq_frame_size) 

        return len(msg)


    @classmethod
    def to_model_update_message(cls, slot, signs, datas, expire_timet=86400*2, version=None):
        msg = ModelUpdateMessage()
        msg.embedding_weight_size = -1
        msg.version = version
        for sign, value in zip(signs, datas):
            item = ModelItem(embedding_weight=value.tobytes(), slot=slot, sign=sign, expire_timet=expire_timet)
            msg.item.append(item)
        return msg
    

    def send_sparse_embs(self, embedding_dict, model_time_ms=-1):
        ret = self.SendSparseEmbeddings(embedding_dict, model_time_ms)
     

    def async_send_sparse_embs(self, embedding_dict, model_time_ms=-1):
        self.emb_send_queue.put((embedding_dict, model_time_ms))


    def SendSparseEmbeddings(self, embedding_dict, model_time_ms=-1):
        send_cnt = dict()
        for emb_config in self.emb_configs:
            emb_name = emb_config["name"]
            emb_dtype = emb_config["dtype"]
            emb_slot = emb_config["slot"]

            emb_data = embedding_dict.get(emb_name, None)
            if emb_data is None:
                print(f"Send Sparse, [{emb_name}] emb not exist, send nothing!!!")
                continue

            keys = emb_data["key"]
            values = emb_data["value"]
            # print('emb_name: ', emb_name, ", key_shape: ", keys.shape, ", value_shape: ", values.shape)
            # 转换 emb 数据格式
            values = convert_embedding_dtype(values)
            # 转换 sign 格式 + slot
            if emb_slot > 0:
                keys = ((emb_slot - 16) << 48) | (keys & ((1 << 48) - 1))

            for s in range(self.emb_btq_shard):
                shard_idx = keys % self.emb_btq_shard == s
                shard_keys = keys[shard_idx].copy()
                shard_values = values[shard_idx]
                shard_size = shard_keys.shape[0]
                # 按 batch 发送，避免消息太大
                for index in range(0, shard_size, self.emb_batch_size):
                    index_end = min(shard_size, index + self.emb_batch_size)
                    batch_keys = shard_keys[index: index_end]
                    batch_values = shard_values[index: index_end]

                    if self.debug_mode and s == 0:
                        print(f'debug send_sparse shard[{s}]'.center(100, '-'))
                        print(f"emb_name: {emb_name}")
                        print(f"emb_dtype: {emb_dtype}")
                        print(f"emb_slot: {emb_slot}")
                        # print(f"emb_data: {emb_data}")
                        print(f"batch_keys: {batch_keys[:10]}")
                        print(f"batch_values shape: {batch_values.shape}")

                    msg = ModelSender.to_model_update_message(emb_slot, batch_keys, batch_values, expire_timet=self.emb_expire_timet,
                                                              version=model_time_ms)
                    BtqClient.produce(f"{self.emb_btq_prefix}{s}", [msg.SerializeToString()])
            send_cnt[emb_name] = keys.shape[0]
        return send_cnt

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
    parser.add_argument("--ckpt", type=str, default='/pub/renlejian/repo/deep-vector-quantization/dvq/lightning_logs/version_18/checkpoints/epoch=38-step=241917.ckpt')

    args = parser.parse_args()
    model_sender_config = {
        "debug_mode": True,
        # "send_step": 100, # never send
        "dense": {
            "send_queue_size": 1,
            "btq_frame_size": 1000000,
            "btq_prefix": "semantic_id_decoder",
            "dnn_yaml_file": "deploy/dnn_model.yaml",
            "dnn_weight_clip": None,
        },
    }
    model_sender = ModelSender(model_sender_config)

    model = VQVAE.load_from_checkpoint(args.ckpt, args=args)
    model_state_dict = model.state_dict()
    model_state_dict_cpu = {f"model.{k}": v.cpu() for k, v in model_state_dict.items()}
    # print(model_state_dict_cpu.keys())
    model_sender.SendDNNModel(model_state_dict_cpu, int(time.time())*1000, False)
    print("done.")
    # time.sleep(3600)


if __name__ == "__main__":
    main()