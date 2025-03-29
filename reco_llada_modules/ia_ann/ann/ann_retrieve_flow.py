#!/usr/bin/env python3
# coding=utf-8

import os
import sys
import copy
import json
import math


class AnnRetrieveFlow():
  def __init__(self):
    self._config = {
      "kess_config": {},
      "ann_pool": {},
      "mq_consumer": {},
      "meta_parser": {},
      "datas": {},
      "buckets": {},
      "server_role": 0
    }

  def set_server_role(self, **kwargs):
    """
    对于在离线分离的情况设置 server_role:
    0 表示在离线一体，1 表示生产索引的离线服务，2 表示提供检索的在线服务
    ------
    `server_role`: [int] 默认为 0
    """
    self._config["server_role"] = kwargs["server_role"] if "server_role" in kwargs else 0
    return copy.deepcopy(self)

  def register_kess(self, **kwargs):
    """
    注册 kess name
    参数配置
    ------
    `kess_name`: [string] kess name，不能缺省
    """
    kess_config = dict()
    kess_config["service_name"] = kwargs["kess_name"]
    self._config["kess_config"] = kess_config
    return copy.deepcopy(self)

  def configure_global_index(self, **kwargs):
    """
    索引池
    参数配置
    ------
    `index_update_interval_second`: [int] 索引定时更新的时间间隔，如果 <=0 则不会定时更新索引，默认值 60. 对于按 version 更新索引的不需要配置
    `ann_index_btq_prefix`: [string] 用于传递 ann 索引的 btq name prefix，只有在离线分离的情况需要配置
    `ann_data_btq_prefix`: [string] 用于同步 ann data的 btq name prefix, 只有在离线分离的情况需要配置
    """
    if "index_update_interval_second" in kwargs:
      self._config["ann_pool"]["index_update_interval_second"] = kwargs["index_update_interval_second"]
    
    if "ann_index_btq_prefix" in kwargs :
      mq_version_consumer = dict()
      mq_version_consumer["ann_index_btq_prefix"] = kwargs["ann_index_btq_prefix"]
      if "ann_data_btq_prefix" in kwargs:
        mq_version_consumer["ann_data_btq_prefix"] = kwargs["ann_data_btq_prefix"]
      mq_version_consumer["type_name"] = "btQueueVersionConsumer" 
      self._config["ann_pool"]["mq_version_consumer"] = mq_version_consumer

    return copy.deepcopy(self)

  def consume_data_from_btq(self, **kwargs):
    """
    从 btq 消费数据 
    参数配置
    ------
    `queue_names`: [string_list] btq topic 列表，不能缺省
    `thread_num`: [int] 针对每个 btq topic 的消费线程数，缺省则值为 1
    """
    mq_consumer = dict()
    mq_consumer["type_name"] = "btqueueMqConsumer"
    mq_consumer["queue_names"] = kwargs["queue_names"]
    thread_num = kwargs["thread_num"] if "thread_num" in kwargs else 1
    mq_consumer["thread_num"] = thread_num
    self._config["mq_consumer"] = mq_consumer
    return copy.deepcopy(self)
  
  def consume_data_from_kafka(self, **kwargs):
    """
    从 kafka 消费数据
    参数配置
    ------
    `topic`: [string] kafka topic, 不能缺省
    `consumer_group`: [string] kafka consumer_group, 不能缺省
    `consume_mode`: [int] kafka consume_mode, 缺省则值为 0
    `thread_num`: [int] 消费线程数，缺省则值为 1
    """
    mq_consumer = dict()
    mq_consumer["type_name"] = "kafkaMqConsumer"
    mq_consumer["topic"] = kwargs["topic"]
    mq_consumer["consumer_group"] = kwargs["consumer_group"]
    consume_mode = kwargs["consume_mode"] if "consume_mode" in kwargs else 0
    mq_consumer["consume_mode"] = consume_mode
    thread_num = kwargs["thread_num"] if "thread_num" in kwargs else 1
    mq_consumer["thread_num"] = thread_num
    self._config["mq_consumer"] = mq_consumer
    return copy.deepcopy(self)

  def parse_data_in_common(self, **kwargs):
    """
    解析中台 ann retrieve 服务通用格式的数据
    参数配置
    ------
    "data_name": [string] 数据类型名，需要与从 btq 消费得到的 UpdateItem 中的 data_type 字段值保持一致，不能缺省
    "id_converter": [string] 对数据中的 id 如何转换，支持 plainIdConverter/kuibaEmbeddingIdConverter/mioEmbeddingIdConverter 三种类型，
                             一般用 plainIdConverter 即可，只有当使用 kuiba 产出的 Ann Retrieve 通用格式的数据时才需要用 kuibaEmbeddingIdConverter，
                             缺省则值为 plainIdConverter 
    "max_item_num": [int] shm_kv 中最多存放的数据条数，不能缺省
    "dim": [int] 数据维度，不能缺省
    "kv_expire_second": [int] 数据在 shm_kv 中的过期时间，不能缺省
    "enable_embedding_dict": [bool] 是否选择 embedding_dict 存数据，默认值为 false，true 时不再使用 shm_kv
    "delay_delete_second": [int] kv 数据失效后延迟删除时间，shm_kv 默认值为 1, embedding_dict 默认值为 30
    """
    self._config["meta_parser"] = {}
    self._config["datas"] = {}

    data_name = kwargs["data_name"]

    self._config["meta_parser"]["type_name"] = "updateItemParser"
    self._config["meta_parser"]["update_by_version"] = False

    kv = dict()
    dim = kwargs["dim"]
    max_item_num = kwargs["max_item_num"]
    kv["max_item_num"] = max_item_num
    kv["dim"] = dim
    kv["cache_expire_second"] = kwargs["kv_expire_second"]
    if "enable_embedding_dict" in kwargs:
      kv["enable_embedding_dict"] = kwargs["enable_embedding_dict"]
    if "delay_delete_second" in kwargs:
      kv["delay_delete_second"] = kwargs["delay_delete_second"]

    data = dict()
    data["kv"] = kv
    id_converter = kwargs["id_converter"] if "id_converter" in kwargs else "plainIdConverter"
    data["online_id_converter"] = {"type_name": id_converter}
    data["offline_id_converter"] = {"type_name": id_converter}

    self._config["datas"][data_name] = data

    build_index = AnnRetrieveBuildIndex(self._config, data_name, dim, max_item_num)
    return copy.deepcopy(build_index)

  def parse_data_in_mio(self, **kwargs):
    """
    解析 mio 格式的数据
    参数配置
    ------
    "data_name": [string] 数据类型名，由用户自定义，例如 user/item/photo，不能缺省
    "slot_id":  [int] 数据在 mio 训练中指定的 slot_id，不能缺省
    "region_id":  [int] 海外数据在 mio 训练中指定的 region_id，可缺省
    "begin_bit": [int] 数据 embedding 的起始元素下标，不能缺省
    "end_bit": [int] 数据 embedding 的结束元素下标，不能缺省
    "max_item_num": [int] shm_kv 中最多存放的数据条数，不能缺省
    "dim": [int] 数据维度，不能缺省
    "kv_expire_second": [int] 数据在 shm_kv 中的过期时间，不能缺省
    "enable_embedding_dict": [bool] 是否选择 embedding_dict 存数据，默认值为 false，true 时不再使用 shm_kv
    "delay_delete_second": [int] kv 数据失效后延迟删除时间，shm_kv 默认值为 1, embedding_dict 默认值为 30
    "enforce_use_slot_form_item": [bool] 是否强制使用 item.slot()作为slot（不从key sign推断），默认值为false
    "parse_data_in_float": [bool] 是否直接用 float 格式 parse embedding, 默认值为 false
    "slot_bits": [int] keysign 的前 slot_bits 直接做为 slot，需要打开 enable_fixed_slot_bits
    "enable_fixed_slot_bits": [bool] 是否直接使用 keysign 前 slot_bits 做为 slot，与 slot_bits 配合使用
    "use_common_index_id_converter": [bool] 是否使用 keysign 使用 CommonIndex Id 格式解析
    "offline_id_converter": [string] 自配置的 offline_id_converter
    "online_id_converter": [string] 自配置的 online_id_converter
    "use_plain_id_converter": [bool] 是否使用 keysign 使用 CommonIndex Id 格式解析
    "enable_kuiba_sign": [bool] 是否按照 kuiba sign 格式提取 slot
    """
    self._config["meta_parser"] = {}
    self._config["datas"] = {}

    data_name = kwargs["data_name"]

    slot_id = kwargs["slot_id"]
    region_id = kwargs.get("region_id", -1)
    if region_id >= 0:
      slot_id = (slot_id << 8) | region_id
      self._config["meta_parser"]["enable_slot_with_region"] = True
    self._config["meta_parser"]["bucket_slot_map"] = {}
    self._config["meta_parser"]["bucket_slot_map"][data_name] = slot_id

    self._config["meta_parser"]["slot_bits_map"] = {}
    slot_bits_map = dict()
    slot_bits_map["begin_bit"] = kwargs["begin_bit"]
    slot_bits_map["end_bit"] = kwargs["end_bit"]
    self._config["meta_parser"]["slot_bits_map"][str(slot_id)] = slot_bits_map

    self._config["meta_parser"]["slot_bits"] = kwargs.get("slot_bits") or 0
    self._config["meta_parser"]["enable_fixed_slot_bits"] = kwargs.get("enable_fixed_slot_bits") or False
    self._config["meta_parser"]["enforce_use_slot_form_item"] = kwargs.get(
      "enforce_use_slot_form_item") or False
    self._config["meta_parser"]["parse_data_in_float"] = kwargs.get(
      "parse_data_in_float") or False
    self._config["meta_parser"]["enable_kuiba_sign"] = kwargs.get("enable_kuiba_sign") or False
    self._config["meta_parser"]["type_name"] = "mioEmbeddingParser"

    kv = dict()
    dim = kwargs["dim"]
    max_item_num = kwargs["max_item_num"]
    kv["max_item_num"] = max_item_num
    kv["dim"] = dim
    kv["cache_expire_second"] = kwargs["kv_expire_second"]
    if "enable_embedding_dict" in kwargs:
      kv["enable_embedding_dict"] = kwargs["enable_embedding_dict"]
    if "delay_delete_second" in kwargs:
      kv["delay_delete_second"] = kwargs["delay_delete_second"]

    data = dict()
    data["kv"] = kv
    if "offline_id_converter" in kwargs and len(kwargs["offline_id_converter"]) > 0 \
     and "online_id_converter" in kwargs and len(kwargs["online_id_converter"]) > 0:
      data["offline_id_converter"] = {"type_name": kwargs["offline_id_converter"]}
      data["online_id_converter"] = {"type_name": kwargs["online_id_converter"]}
    elif "use_common_index_id_converter" in kwargs and kwargs["use_common_index_id_converter"]:
      data["online_id_converter"] = {"type_name": "commonIndexIdConverter"}
      data["offline_id_converter"] = {"type_name": "commonIndexIdConverter"}
    elif "use_plain_id_converter" in kwargs and kwargs["use_plain_id_converter"]:
      data["online_id_converter"] = {"type_name": "plainIdConverter"}
      data["offline_id_converter"] = {"type_name": "plainIdConverter"}
    elif self._config["meta_parser"]["enable_kuiba_sign"]:
      data["online_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}
      data["offline_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}
    else:
      data["online_id_converter"] = {"type_name": "mioEmbeddingIdConverter"}
      data["offline_id_converter"] = {"type_name": "mioEmbeddingIdConverter"}

    self._config["datas"][data_name] = data

    build_index = AnnRetrieveBuildIndex(self._config, data_name, dim, max_item_num)
    return copy.deepcopy(build_index)

  def parse_data_in_nn(self, **kwargs):
    """
    解析 nnembedding 格式数据
    参数配置
    ------
    "data_name": [string] 数据类型名，由用户自定义，不能缺省
    "data_type": [string] 由 photo calc server 生成的数据中的 data type 不能缺省
    "dim": [int] 数据维度，不能缺省
    "begin_bit": [int] 数据 embedding 的起始元素下标, 默认值为 0
    "end_bit": [int] 数据 embedding 的结束元素下标，默认值为 dim
    "embedding_name": [string] 数据 kuaishou::reco::NNEmbeddingembedding 中的变量，默认值为 click_embedding
    "max_item_num": [int] shm_kv 中最多存放的数据条数，不能缺省    
    "kv_expire_second": [int] 数据在 shm_kv 中的过期时间，不能缺省
    "enable_embedding_dict": [bool] 是否选择 embedding_dict 存数据，默认值为 false，true 时不再使用 shm_kv
    "delay_delete_second": [int] kv 数据失效后延迟删除时间，shm_kv 默认值为 1, embedding_dict 默认值为 30
    """
    self._config["meta_parser"] = {}
    self._config["datas"] = {}

    data_name = kwargs["data_name"]
    data_type = kwargs["data_type"]
    embedding_name = kwargs["embedding_name"]

    self._config["meta_parser"]["type_name"] = "nnEmbeddingParser"
    self._config["meta_parser"]["data_type"] = data_type
    self._config["meta_parser"]["embedding_name"] = embedding_name
    if "begin_bit" and "end_bit" in kwargs:
      self._config["meta_parser"]["data_bits_map"] = {}
      data_bits_map = dict()
      data_bits_map["begin_bit"] = kwargs["begin_bit"]
      data_bits_map["end_bit"] = kwargs["end_bit"]
      self._config["meta_parser"]["data_bits_map"][data_type] = data_bits_map

    kv = dict()
    dim = kwargs["dim"]
    max_item_num = kwargs["max_item_num"]
    kv["dim"] = dim
    kv["max_item_num"] = max_item_num
    kv["cache_expire_second"] = kwargs["kv_expire_second"]
    kv["cache_expire_second"] = kwargs["kv_expire_second"]
    if "enable_embedding_dict" in kwargs:
      kv["enable_embedding_dict"] = kwargs["enable_embedding_dict"]
    if "delay_delete_second" in kwargs:
      kv["delay_delete_second"] = kwargs["delay_delete_second"]

    data = dict()
    data["kv"] = kv
    data["online_id_converter"] = {"type_name": "plainIdConverter"}
    data["offline_id_converter"] = {"type_name": "plainIdConverter"}

    self._config["datas"][data_name] = data

    build_index = AnnRetrieveBuildIndex(self._config, data_name, dim, max_item_num)
    return copy.deepcopy(build_index)

  def parse_data_in_mio_concat(self, **kwargs):
    """
    解析 mio 拼接 embedding 格式的数据
    参数配置
    ------
    "data_name": [string] 数据类型名，由用户自定义，例如 user/item/photo，不能缺省
    "slot_id":  [int] 数据在 mio 训练中指定的 slot_id，不能缺省
    "begin_bit": [int] 数据 embedding 的起始元素下标，不能缺省
    "end_bit": [int] 数据 embedding 的结束元素下标，不能缺省
    "max_item_num": [int] shm_kv 中最多存放的数据条数，不能缺省
    "dim": [int] 数据维度，不能缺省
    "kv_expire_second": [int] 数据在 shm_kv 中的过期时间，不能缺省
    "enable_embedding_dict": [bool] 是否选择 embedding_dict 存数据，默认值为 false，true 时不再使用 shm_kv
    "delay_delete_second": [int] kv 数据失效后延迟删除时间，shm_kv 默认值为 1, embedding_dict 默认值为 30
    "enforce_use_slot_form_item" : [bool] 是否强制使用 item.slot()作为slot（不从key sign推断），默认值为false
    "parse_data_in_float": [bool] 是否直接用 float 格式 parse embedding, 默认值为 false
    "slot_bits": [int] keysign 的前 slot_bits 直接做为 slot，需要打开 enable_fixed_slot_bits
    "enable_fixed_slot_bits": [bool] 是否直接使用 keysign 前 slot_bits 做为 slot，与 slot_bits 配合使用
    "use_common_index_id_converter": [bool] 是否使用 keysign 使用 CommonIndex Id 格式解析
    "offline_id_converter": [string] 自配置的 offline_id_converter
    "online_id_converter": [string] 自配置的 online_id_converter
    "enable_kuiba_sign": [bool] 是否按照 kuiba sign 格式提取 slot
    """
    self._config["meta_parser"] = {}
    self._config["datas"] = {}

    data_name = kwargs["data_name"]

    slot_id = kwargs["slot_id"]
    self._config["meta_parser"]["data_slot_map"] = {}
    self._config["meta_parser"]["data_slot_map"][data_name] = slot_id

    self._config["meta_parser"]["data_bits_map"] = {}
    data_bits_map = dict()
    data_bits_map["begin_bit"] = kwargs["begin_bit"]
    data_bits_map["end_bit"] = kwargs["end_bit"]
    self._config["meta_parser"]["data_bits_map"][data_name] = data_bits_map

    self._config["meta_parser"]["enforce_use_slot_form_item"] = kwargs.get(
      "enforce_use_slot_form_item") or False
    self._config["meta_parser"]["slot_bits"] = kwargs.get("slot_bits") or 0
    self._config["meta_parser"]["enable_fixed_slot_bits"] = kwargs.get("enable_fixed_slot_bits") or False
    self._config["meta_parser"]["parse_data_in_float"] = kwargs.get(
      "parse_data_in_float") or False
    self._config["meta_parser"]["enable_kuiba_sign"] = kwargs.get("enable_kuiba_sign") or False
    self._config["meta_parser"]["type_name"] = "mioConcatEmbeddingParser"

    kv = dict()
    dim = kwargs["dim"]
    max_item_num = kwargs["max_item_num"]
    kv["max_item_num"] = max_item_num
    kv["dim"] = dim
    kv["cache_expire_second"] = kwargs["kv_expire_second"]
    if "enable_embedding_dict" in kwargs:
      kv["enable_embedding_dict"] = kwargs["enable_embedding_dict"]
    if "delay_delete_second" in kwargs:
      kv["delay_delete_second"] = kwargs["delay_delete_second"]

    data = dict()
    data["kv"] = kv
    if "offline_id_converter" in kwargs and len(kwargs["offline_id_converter"]) > 0 \
     and "online_id_converter" in kwargs and len(kwargs["online_id_converter"]) > 0:
      data["offline_id_converter"] = {"type_name": kwargs["offline_id_converter"]}
      data["online_id_converter"] = {"type_name": kwargs["online_id_converter"]}
    elif "use_common_index_id_converter" in kwargs and kwargs["use_common_index_id_converter"]:
      data["online_id_converter"] = {"type_name": "commonIndexIdConverter"}
      data["offline_id_converter"] = {"type_name": "commonIndexIdConverter"}
    elif self._config["meta_parser"]["enable_kuiba_sign"]:
      data["online_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}
      data["offline_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}
    else:
      data["online_id_converter"] = {"type_name": "mioEmbeddingIdConverter"}
      data["offline_id_converter"] = {"type_name": "mioEmbeddingIdConverter"}

    self._config["datas"][data_name] = data

    build_index = AnnRetrieveBuildIndex(self._config, data_name, dim, max_item_num)
    return copy.deepcopy(build_index)

  def parse_data_in_kuiba(self, **kwargs):
    """
    解析 kuiba 格式的数据
    参数配置
    ------
    "data_name": [string] 数据类型名，由用户自定义，例如 user/item/photo，不能缺省
    "key_type":  [int] 数据在 kuiba 中指定的 key_type，不能缺省
    "max_item_num": [int] shm_kv 中最多存放的数据条数，不能缺省
    "dim": [int] 数据维度，不能缺省
    "kv_expire_second": [int] 数据在 shm_kv 中的过期时间，不能缺省
    "enable_embedding_dict": [bool] 是否选择 embedding_dict 存数据，默认值为 false，true 时不再使用 shm_kv
    "delay_delete_second": [int] kv 数据失效后延迟删除时间，shm_kv 默认值为 1, embedding_dict 默认值为 30
    """
    self._config["meta_parser"] = {}
    self._config["datas"] = {}

    data_name = kwargs["data_name"]

    self._config["meta_parser"]["bucket_slot_map"] = {}
    self._config["meta_parser"]["bucket_slot_map"][data_name] = kwargs["key_type"]
    self._config["meta_parser"]["type_name"] = "kuibaDNNParser"

    kv = dict()
    dim = kwargs["dim"]
    max_item_num = kwargs["max_item_num"]
    kv["max_item_num"] = max_item_num
    kv["dim"] = dim
    kv["cache_expire_second"] = kwargs["kv_expire_second"]
    if "enable_embedding_dict" in kwargs:
      kv["enable_embedding_dict"] = kwargs["enable_embedding_dict"]
    if "delay_delete_second" in kwargs:
      kv["delay_delete_second"] = kwargs["delay_delete_second"]

    data = dict()
    data["kv"] = kv
    data["online_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}
    data["offline_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}

    self._config["datas"][data_name] = data

    build_index = AnnRetrieveBuildIndex(self._config, data_name, dim, max_item_num)
    return copy.deepcopy(build_index)

  def parse_data_in_kuiba_calc(self, **kwargs):
    """
    解析 kuiba calc service 产出格式的数据
    参数配置
    ------
    "data_name": [string] 数据类型名，由用户自定义，例如 user/item/photo，不能缺省
    "tensor_name": [string] 由 kuiba calc service 生成的数据中的 tensor name，不能缺省
    "max_item_num": [int] shm_kv 中最多存放的数据条数，不能缺省
    "dim": [int] 数据维度，不能缺省
    "kv_expire_second": [int] 数据在 shm_kv 中的过期时间，不能缺省
    "enable_embedding_dict": [bool] 是否选择 embedding_dict 存数据，默认值为 false，true 时不再使用 shm_kv
    "delay_delete_second": [int] kv 数据失效后延迟删除时间，shm_kv 默认值为 1, embedding_dict 默认值为 30
    """
    self._config["meta_parser"] = {}
    self._config["datas"] = {}

    data_name = kwargs["data_name"]

    self._config["meta_parser"]["type_name"] = "kuibaCalcServiceParser"
    self._config["meta_parser"]["datas"] = {}
    self._config["meta_parser"]["datas"][kwargs["tensor_name"]] = data_name

    kv = dict()
    dim = kwargs["dim"]
    max_item_num = kwargs["max_item_num"]
    kv["max_item_num"] = max_item_num
    kv["dim"] = dim
    kv["cache_expire_second"] = kwargs["kv_expire_second"]
    if "enable_embedding_dict" in kwargs:
      kv["enable_embedding_dict"] = kwargs["enable_embedding_dict"]
    if "delay_delete_second" in kwargs:
      kv["delay_delete_second"] = kwargs["delay_delete_second"]

    data = dict()
    data["kv"] = kv
    data["online_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}
    data["offline_id_converter"] = {"type_name": "kuibaEmbeddingIdConverter"}

    self._config["datas"][data_name] = data

    build_index = AnnRetrieveBuildIndex(self._config, data_name, dim, max_item_num)
    return copy.deepcopy(build_index)


class AnnRetrieveBuildIndex():
  def __init__(self, config, data_name, dim, max_item_num):
    self._config = config
    self._data_name = data_name
    self._dim = dim
    self._bucket_name = ""
    self._space = ""
    self._max_item_num = max_item_num
    self._weight = None

  def build_index_plugin(self, bucket, kwargs):
    if "booster_item_attr" in kwargs or "booster_dynamic_item_attr" in kwargs:
      booster = dict()
      booster["item_attr"] = kwargs["booster_item_attr"] if "booster_item_attr" in kwargs else ""
      booster["dynamic_item_attr"] = kwargs["booster_dynamic_item_attr"] if "booster_dynamic_item_attr" in kwargs else ""
      bucket["booster"] = booster
    
    if "integrator_data_type" in kwargs:
      integrator = dict()
      integrator["data_type"] = kwargs["integrator_data_type"]
      integrator["item_attr_for_id"] = kwargs["integrator_item_attr_for_id"]
      integrator["stretch_factor"] = kwargs["integrator_strech_factor"] if "strech_factor" in kwargs else 1.0
      bucket["integrator"] = integrator
    
    if "filter_by_keys_from_inverted_list" in kwargs:
      bucket["filter_by_keys_from_inverted_list"] = kwargs["filter_by_keys_from_inverted_list"]
    
    if "arranger_item_attr" in kwargs:
      arranger = dict()
      arranger["item_attr"] = kwargs["arranger_item_attr"]
      arranger["limit"] = kwargs["arranger_limit"] if "arranger_limit" in kwargs else 0
      if "arranger_default_score" in kwargs:
        arranger["arranger_default_score"] = kwargs["arranger_default_score"]
      bucket["arranger"] = arranger

  def build_faiss_index(self, **kwargs):
    """
    构建 faiss 索引
    参数配置
    ------
    `bucket_name`: [string] 对数据建 faiss 索引后，用 bucket name 对其命名，由用户自定义，例如 user_index/item_index/photo_index，不能缺省
    `space`: [string] 计算距离时采用的方式，支持 l2/ip/cosine 三种类型，不能缺省
    `nlist`: [int] faiss 算法参数，聚类个数，缺省则会根据 kv 的 max_item_num 生成，如果实际索引量与 kv 的 max_item_num 相差很大，则不要缺省
    `batch_size`: [int] faiss 算法参数，参与聚类训练的样本数，缺省则值为 256*nlist
    `nprobe`: [int] faiss 算法参数，search 时搜索几个聚类，缺省则值为 max(nlist/200, 10)
    `shard_num`: [int] faiss 算法参数，参数值大于 1 时会采用 IndexShards 功能，缺省则值为 1
    `threaded`: [bool] faiss 算法参数，当开启 IndexShards 功能时，是否用多线程操作各个 shard，缺省则值为 false
    `max_index_size`: [int] 索引量上限，当需建索引的 item 数量超过 max_index_size 时会被截断，缺省则值为 0，表示不设置上限
    `min_index_size`: [int] 索引量下限，当需建索引的 item 数量小于 min_index_size 时，索引更新会失败，缺省则值为 0，表示不设置下限
    `enable_empty_index`: [bool] 是否允许空 index 存在，打开这个选项会自动调参, 默认值是 false
    `threshold_kconf_key`: [string] kconf key，用于读取 ann 算法参数配置，可缺省
    """
    self._config["buckets"] = {}
    parameter = dict()
    space = kwargs["space"]
    self._space = space
    parameter["space"] = space
    parameter["dim"] = self._dim

    nlist = kwargs["nlist"] if "nlist" in kwargs else int(math.sqrt(self._max_item_num))
    nprobe = kwargs["nprobe"] if "nprobe" in kwargs else max(int(nlist / 200), 3)
    batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 256 * nlist
    shard_num = kwargs["shard_num"] if "shard_num" in kwargs else 1
    threaded = kwargs["threaded"] if "threaded" in kwargs else False

    threshold = dict()
    threshold["index_type"] = "ivf"
    threshold["batch_size"] = batch_size
    threshold["nlist"] = nlist
    threshold["nprobe"] = nprobe
    threshold["shard_num"] = shard_num
    threshold["threaded"] = threaded

    ann = dict()
    ann["type_name"] = "faissAnnFloat"
    ann["parameter"] = parameter
    ann["threshold"] = threshold
    ann["max_index_size"] = kwargs["max_index_size"] if "max_index_size" in kwargs else 0
    ann["min_index_size"] = kwargs["min_index_size"] if "min_index_size" in kwargs else 0

    bucket = dict()
    bucket["data_type"] = self._data_name
    bucket["ann"] = ann
    if "enable_empty_index" in kwargs:
      bucket["enable_empty_index"] = kwargs["enable_empty_index"]

    if "threshold_kconf_key" in kwargs:
      threshold = dict()
      threshold["threshold_kconf_key"] = kwargs["threshold_kconf_key"]
      bucket["dynamic_threshold"] = threshold
    
    self.build_index_plugin(bucket, kwargs)

    bucket_name = kwargs["bucket_name"]
    self._bucket_name = bucket_name

    self._config["buckets"][bucket_name] = bucket
    return copy.deepcopy(self)

  def build_scann_index(self, **kwargs):
    """
    构建 scann 索引
    参数配置
    ------
    `bucket_name`: [string] 对数据建 faiss 索引后，用 bucket name 对其命名，由用户自定义，例如 user_index/item_index/photo_index，不能缺省
    `space`: [string] 计算距离时采用的方式，支持 l2/ip/cosine 三种类型，不能缺省
    `final_neighbors_num`: [int] 最终要召回的 topK，不能缺省
    `leaves_num`: [int] 分片数量，不能缺省
    `leaves_to_search`: [int] ann搜索的分片数量，越大准确率越高，不能缺省
    `pre_reorder_neighbors_num`: [int] 重排的近邻数量，越大准确率越高，不能缺省
    `training_sample_size`: [int] 训练embedding分片核心的采样数量，越大越准，索引构建时间越长，不能缺省
    `anisotropic_quantization_threshold`: [float] 训练的参数，与数据分布有关，距离超过这个值的样本会被丢掉，不能缺省
    `max_index_size`: [int] 索引量上限，当需建索引的 item 数量超过 max_index_size 时会被截断，缺省则值为 0，表示不设置上限
    `min_index_size`: [int] 索引量下限，当需建索引的 item 数量小于 min_index_size 时，索引更新会失败，缺省则值为 0，表示不设置下限
    `enable_empty_index`: [bool] 是否允许空 index 存在，打开这个选项会自动调参, 默认值为 false
    `threshold_kconf_key`: [string] kconf key，用于读取 ann 算法参数配置，可缺省
    """
    self._config["buckets"] = {}
    parameter = dict()
    space = kwargs["space"]
    self._space = space
    parameter["space"] = space
    parameter["dim"] = self._dim

    threshold = dict()
    threshold["final_neighbors_num"] = kwargs["final_neighbors_num"]
    threshold["leaves_num"] = kwargs["leaves_num"]
    threshold["leaves_to_search"] = kwargs["leaves_to_search"]
    threshold["pre_reorder_neighbors_num"] = kwargs["pre_reorder_neighbors_num"]
    threshold["training_sample_size"] = kwargs["training_sample_size"]
    threshold["anisotropic_quantization_threshold"] = kwargs["anisotropic_quantization_threshold"]

    ann = dict()
    ann["type_name"] = "scannAnnFloat"
    ann["parameter"] = parameter
    ann["threshold"] = threshold
    ann["max_index_size"] = kwargs["max_index_size"] if "max_index_size" in kwargs else 0
    ann["min_index_size"] = kwargs["min_index_size"] if "min_index_size" in kwargs else 0

    bucket = dict()
    bucket["data_type"] = self._data_name
    bucket["ann"] = ann
    if "enable_empty_index" in kwargs:
      bucket["enable_empty_index"] = kwargs["enable_empty_index"]

    if "threshold_kconf_key" in kwargs:
      threshold = dict()
      threshold["threshold_kconf_key"] = kwargs["threshold_kconf_key"]
      bucket["dynamic_threshold"] = threshold
    
    self.build_index_plugin(bucket, kwargs)

    bucket_name = kwargs["bucket_name"]
    self._bucket_name = bucket_name

    self._config["buckets"][bucket_name] = bucket
    return copy.deepcopy(self)

  def build_knn_index(self, **kwargs):
    """
    构建 scann 索引
    参数配置
    ------
    `bucket_name`: [string] 对数据建 faiss 索引后，用 bucket name 对其命名，由用户自定义，例如 user_index/item_index/photo_index，不能缺省
    `space`: [string] 计算距离时采用的方式，仅支持 ip
    `threshold_kconf_key`: [string] kconf key，用于读取 ann 算法参数配置，可缺省
    `weight`: [list[float]] 多目标 xtr 线性拟合系数, 如目标为 ctr、ltr, 那么 weight 就配置为 [w1, w2]
    `batch_size`: [int] requst batch 计算的最大数量
    `capacity`: [int] requst batch 计算的最大数量, 缺省为 1000000 
    `target_dim`: [int] 单目标target item embedding 维度
    `top_k`: [int] knn top_k
    `topk_batch_size`: [int] topk batch 计算的最大数量，通常无需调整
    `cuda_stream_num_per_device`: [int] 底层配置，通常无需调整
    `graph_type`: [int]计算类型 1、多目标 2、单目标
    """
    self._config["buckets"] = {}
    parameter = dict()
    space = kwargs["space"]
    self._space = space
    parameter["space"] = space
    parameter["dim"] = self._dim
    self._weight = kwargs["weight"]

    threshold = dict()
    threshold["weight"] = kwargs["weight"]
    threshold["batch_size"] = kwargs["batch_size"]
    threshold["capacity"] = kwargs["capacity"]
    threshold["target_dim"] = kwargs["target_dim"]
    threshold["top_k"] = kwargs["top_k"]

    if "topk_batch_size" in kwargs:
      threshold["topk_batch_size"] = kwargs["topk_batch_size"]
    if "cuda_stream_num_per_device" in kwargs:
      threshold["cuda_stream_num_per_device"] = kwargs["cuda_stream_num_per_device"]
    if "graph_type" in kwargs:
      threshold["graph_type"] = kwargs["graph_type"]

    ann = dict()
    ann["type_name"] = "fullCalcKnnFloat"
    ann["parameter"] = parameter
    ann["threshold"] = threshold
    ann["max_index_size"] = kwargs["max_index_size"] if "max_index_size" in kwargs else 0
    ann["min_index_size"] = kwargs["min_index_size"] if "min_index_size" in kwargs else 0

    bucket = dict()
    bucket["data_type"] = self._data_name
    bucket["ann"] = ann
    if "enable_empty_index" in kwargs:
      bucket["enable_empty_index"] = kwargs["enable_empty_index"]

    if "threshold_kconf_key" in kwargs:
      threshold = dict()
      threshold["threshold_kconf_key"] = kwargs["threshold_kconf_key"]
      bucket["dynamic_threshold"] = threshold
    
    self.build_index_plugin(bucket, kwargs)

    bucket_name = kwargs["bucket_name"]
    self._bucket_name = bucket_name

    self._config["buckets"][bucket_name] = bucket
    return copy.deepcopy(self)

  def filter_index_by_white_list(self, **kwargs):
    """
    通过 redis 白名单过滤索引
    参数配置
    ------
    `redis_key`: [string] 白名单的 redis key，不能缺省
    """
    index_filter = dict()
    index_filter["type_name"] = "redisIndexFilter"
    index_filter["redis_key"] = kwargs["redis_key"]
    if "update_batch_interval_s" in kwargs:
      index_filter["update_batch_interval_s"] = kwargs["update_batch_interval_s"]
    self._config["buckets"][self._bucket_name]["index_filter"] = index_filter
    return copy.deepcopy(self)

  def filter_index_by_item_attr(self, **kwargs):
    """
    通过中台 common index 的 item_attr 过滤索引
    参数配置
    ------
    `compare_to`: [int/float/string] 比较的对象值，不能缺省 
    `item_attr`: [string] 使用 common index 中的哪个 item_attr 与 compare_to 做比较，不能缺省
    `remove_if`: [string] 支持的比较操作符，包括 >= <= > < == !=，不能缺省
    `remove_if_attr_missing`: [bool] 如果指定的 item_attr 不存在，是否过滤，不能缺省
    `item_type`: [int] common index 内的 item_type
    """
    index_filter = dict()
    index_filter["type_name"] = "itemAttrFilter"
    index_filter["compare_to"] = kwargs["compare_to"]
    index_filter["item_attr"] = kwargs["item_attr"]
    index_filter["remove_if"] = kwargs["remove_if"]
    index_filter["remove_if_attr_missing"] = kwargs["remove_if_attr_missing"]
    self._config["buckets"][self._bucket_name]["index_filter_by_common_index"] = index_filter
    if "item_type" in kwargs:
      self._config["buckets"][self._bucket_name]["item_type"] = kwargs["item_type"]
    return copy.deepcopy(self)

  def filter_index_by_lua_script(self, **kwargs):
    """
    通过中台 common index 和 lua 脚本的方式过滤索引
    参数配置
    ------
    `import_item_attr`: [string_list] lua 脚本中需要用到 common index 中的哪些 item_attr，不能缺省
    `item_remove_check_func`: [string] lua 脚本中的函数名，不能缺省
    `lua_script`: [string] lua 脚本，当脚本返回 true 时做过滤，不能缺省
    `remove_if_lua_fail`: [bool] 当 lua 脚本执行失败时是否做过滤，不能缺省
    `item_type`: [int] common index 内的 item_type
    """
    index_filter = dict()
    index_filter["type_name"] = "itemLuaFilter"
    index_filter["import_item_attr"] = kwargs["import_item_attr"]
    index_filter["item_remove_check_func"] = kwargs["item_remove_check_func"]
    index_filter["lua_script"] = kwargs["lua_script"]
    index_filter["remove_if_lua_fail"] = kwargs["remove_if_lua_fail"]
    self._config["buckets"][self._bucket_name]["index_filter_by_common_index"] = index_filter
    if "item_type" in kwargs:
      self._config["buckets"][self._bucket_name]["item_type"] = kwargs["item_type"]
    return copy.deepcopy(self)

  def filter_index_by_keys_from_inverted_list(self, filter_by_keys_from_inverted_list):
    """
    使用通用倒排中的 keys 过滤 embedding
    ------
    `filter_by_keys_from_inverted_list`: [string] 通用倒排中 keys 对应的名称
    """
    self._config["buckets"][self._bucket_name][
      "filter_by_keys_from_inverted_list"] = filter_by_keys_from_inverted_list
    return copy.deepcopy(self)

  def retrieve_from(self, **kwargs):
    """
    设置召回模式，例如 i2i u2i u2u
    参数配置
    ------
    `dest_bucket`: [AnnRetrieveBuildIndex] 从哪个 bucket 召回结果，不能缺省
    `cache_expire_second`: [int] 缓存召回结果的过期时间，缺省则值为 3600
    `cache_max_item_num`: [int] 缓存召回结果的个数上限，缺省则值为 10000
    `enable_precision_eval`: [bool] 是否开启在线计算召回准确率，不能缺省
    `enable_auto_calc`: [bool] 是否开启缓存，不能缺省
    """

    self._config["buckets"] = {}
    dest_bucket = kwargs["dest_bucket"]
    try:
      enable_auto_calc = kwargs["enable_auto_calc"]
      if enable_auto_calc:
        self._config["auto_calc"] = []
        one_cache = dict()
        one_cache["src_data_type"] = self._data_name
        one_cache["dest_bucket"] = dest_bucket._bucket_name
        cache_expire_second = kwargs["cache_expire_second"] if "cache_expire_second" in kwargs else 3600
        cache_max_item_num = kwargs["cache_max_item_num"] if "cache_max_item_num" in kwargs else 10000
        one_cache["expire_second"] = cache_expire_second
        one_cache["max_item_num"] = cache_max_item_num
        one_cache["recalc_interval_second"] = 0
        self._config["auto_calc"].append(one_cache)
    except Exception as e:
      print(e, " is missing, please specify True(use auto calc) or False")

    try:
      enable_precision_eval = kwargs["enable_precision_eval"]
      if enable_precision_eval:
        self._config["precision_eval"] = []
        one_eval = dict()
        one_eval["src_data_type"] = self._data_name
        one_eval["dest_bucket"] = dest_bucket._bucket_name
        one_eval["space"] = dest_bucket._space
        if dest_bucket._weight:
          one_eval["weight"] = dest_bucket._weight
          one_eval["dim"] = int(dest_bucket._dim / len(dest_bucket._weight))
        self._config["precision_eval"].append(one_eval)
    except Exception as e:
      print(e, " is missing, please specify True(use precision eval) or False")

    self.merge_config(dest_bucket)
    return copy.deepcopy(self)

  def merge_config(self, dest_bucket):
    """
    合并两个 ann server 的配置
    参数配置
    ------
    `dest_bucket`: [AnnRetrieveBuildIndex] 跟哪个 ann server 的配置做合并
    """
    self._config["datas"].update(dest_bucket._config["datas"])
    self._config["buckets"].update(dest_bucket._config["buckets"])
    if "bucket_slot_map" in self._config["meta_parser"] and "bucket_slot_map" in dest_bucket._config[
      "meta_parser"]:
      self._config["meta_parser"]["bucket_slot_map"].update(
        dest_bucket._config["meta_parser"]["bucket_slot_map"])
    if "slot_bits_map" in self._config["meta_parser"] and "slot_bits_map" in dest_bucket._config[
      "meta_parser"]:
      self._config["meta_parser"]["slot_bits_map"].update(dest_bucket._config["meta_parser"]["slot_bits_map"])
    if "data_slot_map" in self._config["meta_parser"] and "data_slot_map" in dest_bucket._config[
      "meta_parser"]:
      self._config["meta_parser"]["data_slot_map"].update(dest_bucket._config["meta_parser"]["data_slot_map"])
    if "data_bits_map" in self._config["meta_parser"] and "data_bits_map" in dest_bucket._config[
      "meta_parser"]:
      self._config["meta_parser"]["data_bits_map"].update(dest_bucket._config["meta_parser"]["data_bits_map"])
    if "datas" in self._config["meta_parser"] and "datas" in dest_bucket._config["meta_parser"]:
      self._config["meta_parser"]["datas"].update(dest_bucket._config["meta_parser"]["datas"])

    if "auto_calc" in self._config:
      if "auto_calc" in dest_bucket._config:
        self._config["auto_calc"].extend(dest_bucket._config["auto_calc"])
    else:
      if "auto_calc" in dest_bucket._config:
        self._config["auto_calc"] = dest_bucket._config["auto_calc"]

    if "precision_eval" in self._config:
      if "precision_eval" in dest_bucket._config:
        self._config["precision_eval"].extend(dest_bucket._config["precision_eval"])
    else:
      if "precision_eval" in dest_bucket._config:
        self._config["precision_eval"] = dest_bucket._config["precision_eval"]

    return copy.deepcopy(self)

  def print_config(self):
    """
    按 json 格式输出配置
    """
    print(json.dumps(self._config, sort_keys=False, indent=2))

  def get_config(self):
    """
    直接返回配置
    """
    return self._config

  def integrate_with(self, data_type, data_key_in_common_index, item_type=0, stretch_factor=1.0):
    """
    只对 item 类型的 embedding 起作用，需要有对应的通用索引
    ------
    `data_type`: [string] 要累加的 embedding 对应的 data type, fm retr 中一般是 author_id 或者 aid
    `data_key_in_common_index`: [string] item 对应的 aid 要从通用索引的哪个字段读取，一般为 author_id 或者 author__id
    `item_type`: [int] common index 内的 item_type
    `stretch_factor`: [float] 对要加和的 embedding 做拉伸，对 embedding 每一位乘上 stretch_factor
    """
    integrator = dict()
    integrator["data_type"] = data_type
    integrator["item_attr_for_id"] = data_key_in_common_index
    integrator["stretch_factor"] = stretch_factor
    if item_type:
      self._config["buckets"][self._bucket_name]["item_type"] = item_type
    self._config["buckets"][self._bucket_name]["integrator"] = integrator
    return copy.deepcopy(self)