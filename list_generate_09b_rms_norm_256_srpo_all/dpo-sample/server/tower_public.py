#!/usr/bin/env python3
# coding=utf-8

import os
import sys
import logging
import copy
import json
import yaml
import argparse
import base64
import collections
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(filename)s:%(lineno)s %(message)s",
)
current_dir = os.path.dirname(__file__)

from dragonfly.common_leaf_dsl import OfflineRunner, LeafService, LeafFlow
from dragonfly.common_leaf_util import ArgumentError, LogicError
from dragonfly.ext.common.common_api_mixin import CommonApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin


class TowerPredictFlowBase(
    LeafFlow,
    KuibaApiMixin,
    MioApiMixin,
    OfflineApiMixin,
    EmbedCalcApiMixin,
    GsuApiMixin,
):
    pass


def get_attrs_from_kuiba_parameter_config(conf):
    attrs = set()
    for name, c in conf.items():
        for e in c["attrs"]:
            attrs.update(e["attr"])

    ret = [e for e in attrs]
    ret.sort()
    return ret


# ----------------------------------------- #
#       model with mio feature
# ----------------------------------------- #

# load Resources
ModelConfigWithKsSignFeature = collections.namedtuple(
    "ModelConfigWithKsSignFeature",
    [
        "graph",
        "outputs",
        "slots_config",
        "param",
        "common_slots",
        "non_common_slots",
        "feature_list",
    ],
)


def load_feature_list_sign(filename):
    ret = set()
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            for field in line.strip().split(","):
                parts = field.strip().split("=")
                assert len(parts) == 2, "Unsupported format: " + line.strip()

                if parts[0].strip() != "class":
                    # ignore unknown field
                    continue

                ret.add(parts[1].strip())
    return list(sorted(ret))


def load_mio_tf_tower_model_with_ks_sign_feature(
    config_root_dir, model_name, predict_type
):
    model_dir = os.path.join(
        config_root_dir, "models", model_name, predict_type
    )

    with open(os.path.join(model_dir, "dnn_model.yaml")) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(model_dir, "graph.pb"), "rb") as f:
        base64_graph = base64.b64encode(f.read()).decode("ascii")
        graph = "base64://" + base64_graph

    feature_list = load_feature_list_sign(
        os.path.join(
            config_root_dir, "models", model_name, "feature_list_sign.txt"
        )
    )

    graph_tensor_mapping = dnn_model["graph_tensor_mapping"]
    q_names = dnn_model["q_names"].split(" ")
    extra_preds = q_names
    outputs = [
        (extra_pred, graph_tensor_mapping[q_name])
        for extra_pred, q_name in zip(extra_preds, q_names)
    ]
    param = [
        param
        for param in dnn_model["param"]
        if param.get("send_to_online", True)
    ]

    slots_config = dnn_model["embedding"]["slots_config"]
    common_slots = set()
    non_common_slots = set()
    for c in slots_config:
        slots = map(int, str(c["slots"]).split(" "))
        if c.get("common", False):
            common_slots.update(slots)
        else:
            non_common_slots.update(slots)

    return ModelConfigWithKsSignFeature(
        graph,
        outputs,
        slots_config,
        param,
        common_slots,
        non_common_slots,
        feature_list,
    )


# ----------------------------------------- #
#       model with kuiba feature
# ----------------------------------------- #

ModelConfigWithKuibaAttr = collections.namedtuple(
    "ModelConfigWithKuibaAttr",
    [
        "graph",
        "outputs",
        "slots_config",
        "param",
        "common_slots",
        "non_common_slots",
        "pid_slots",
        "common_parameter_config",
        "non_common_parameter_config",
    ],
)


def load_mio_tf_tower_model_with_kuiba_attr(
    config_root_dir, model_name, predict_type
):
    model_dir = os.path.join(
        config_root_dir, "models", model_name, predict_type
    )

    with open(os.path.join(model_dir, "dnn_model.yaml")) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(model_dir, "graph.pb"), "rb") as f:
        base64_graph = base64.b64encode(f.read()).decode("ascii")
        graph = "base64://" + base64_graph

    with open(
        os.path.join(
            config_root_dir, "models", model_name, "parameter_config.json"
        )
    ) as f:
        parameter_config = json.load(f)

    graph_tensor_mapping = dnn_model["graph_tensor_mapping"]
    q_names = dnn_model["q_names"].split(" ")
    extra_preds = q_names
    outputs = [
        (extra_pred, graph_tensor_mapping[q_name])
        for extra_pred, q_name in zip(extra_preds, q_names)
    ]
    param = [
        param
        for param in dnn_model["param"]
        if param.get("send_to_online", True)
    ]

    slots_config = dnn_model["embedding"]["slots_config"]
    common_slots = set()
    non_common_slots = set()
    pid_slots = set()
    for c in slots_config:
        slots = map(int, str(c["slots"]).split(" "))
        if c.get("common", False):
            common_slots.update(slots)
        else:
            non_common_slots.update(slots)

    common_parameter_config = dict()
    non_common_parameter_config = dict()
    for name, c in parameter_config.items():
        attrs = c["attrs"]
        assert len(attrs) == 1
        if attrs[0]["converter"] == "combine":
            converter_args = attrs[0]["converter_args"]
            attr = [
                *converter_args["left"].keys(),
                *converter_args["right"].keys(),
            ]
        else:
            attr = attrs[0]["attr"]

        slot_id = attrs[0]["key_type"]

        if slot_id in common_slots:
            common_parameter_config[name] = c
        else:
            non_common_parameter_config[name] = c
            if len(attr) == 1 and attr[0] == "pId":
                pid_slots.add(slot_id)

    return ModelConfigWithKuibaAttr(
        graph,
        outputs,
        slots_config,
        param,
        common_slots,
        non_common_slots,
        pid_slots,
        common_parameter_config,
        non_common_parameter_config,
    )


def load_mio_tf_tower_model(
    config_root_dir, model_name, predict_type, feature_type
):
    if feature_type == "kuiba":
        logging.debug(
            f"load_mio_tf_tower_model_with_kuiba_attr : {predict_type} ..."
        )
        return load_mio_tf_tower_model_with_kuiba_attr(
            config_root_dir, model_name, predict_type
        )
    else:
        logging.debug(
            f"load_mio_tf_tower_model_with_ks_sign_feature : {predict_type} ..."
        )
        return load_mio_tf_tower_model_with_ks_sign_feature(
            config_root_dir, model_name, predict_type
        )


def get_mio_slot_config_common_outputs(model_config):
    outputs = list()
    for c in model_config.slots_config:
        if c.get("common", False):
            outputs.append(c["input_name"])
    return outputs


def get_mio_slot_config_item_outputs(model_config):
    outputs = list()
    for c in model_config.slots_config:
        if not c.get("common", False):
            outputs.append(c["input_name"])
    return outputs


def get_cur_work_dir_mio_slot_config(
    predict_type="user_predict",
    feature_type="mio",
    common=True,
    config_root_dir="",
    model_name="",
):
    model_config = load_mio_tf_tower_model(
        config_root_dir, model_name, predict_type, feature_type
    )
    return (
        get_mio_slot_config_common_outputs(model_config)
        if common
        else get_mio_slot_config_item_outputs(model_config)
    )


def get_photo_store_kconf_key(kwargs):
    index_service = kwargs.get("index_service", None)
    if None:
        return ""
    if index_service in set(
        ["grpc_hotPhotoInfoService", "grpc_hotPhotoInfoServiceOffline"]
    ):
        return "reco.arch.hotPhotoStoreConfigForTower"
    if index_service in set(["grpc_hotPhotoInfoServiceExp"]):
        return "reco.arch.hotPhotoStoreConfigForTowerExp"
    if index_service in set(["grpc_hotPhotoInfoServiceTnuExt"]):
        return "reco.arch.hotPhotoStoreConfigForTowerTnuExt"
    if index_service in set(["grpc_eyesHotPhotoInfoService"]):
        return "reco.arch.eyesHotPhotoStoreConfigForTower"
    if index_service in set(["grpc_kuibaLiveInfoService"]):
        return ""
    if index_service in set(["grpc_kuibaPhotoInfoService6s"]):
        return ""
    if True:
        raise ArgumentError(
            f"未知的索引服务 {index_service}, 请配置相应的 photo store kconf 配置"
        )


class SimTowerApi:
    ID = "sim_hetu_tags"
    ID_ATTR = "sim_hetu_tags_attr"
    HETU_TAG_ATTR = "hetu_tags"

    @staticmethod
    def is_sim_tower(kwargs):
        return "sim_hetu_tags" in kwargs

    @staticmethod
    def add_sim_tower_item_attrs_from_request(orig_attrs):
        attrs = set()
        if orig_attrs:
            for e in orig_attrs:
                attrs.add(e)
        ret = copy.deepcopy(orig_attrs) if orig_attrs else list()
        if SimTowerApi.HETU_TAG_ATTR not in attrs:
            ret.append(SimTowerApi.HETU_TAG_ATTR)
        return ret

    def add_hetu_tag_0(tag_list):
        tag_set = set()
        for v in tag_list:
            tag_set.add(v)
        if 0 not in tag_set:
            tag_set.add(0)
        tags = [e for e in tag_set]
        tags.sort()
        return tags


def update_leaf_service_common_attrs_from_request(service, extra_attrs):
    common_attrs_from_request = (
        copy.deepcopy(service.common_attrs_from_request)
        if service.common_attrs_from_request
        else list()
    )
    for e in extra_attrs:
        if e not in common_attrs_from_request:
            common_attrs_from_request.append(e)
    service.common_attrs_from_request = common_attrs_from_request


def update_leaf_service_item_attrs_from_request(service, extra_attrs):
    item_attrs_from_request = (
        copy.deepcopy(service.item_attrs_from_request)
        if service.item_attrs_from_request
        else list()
    )
    for e in extra_attrs:
        if e not in item_attrs_from_request:
            item_attrs_from_request.append(e)
    service.item_attrs_from_request = item_attrs_from_request


def ignore_ending_processor_outputs(outputs):
    if not LeafService.IGNORE_UNUSED_ATTR:
        LeafService.IGNORE_UNUSED_ATTR = list()
    for e in outputs:
        LeafService.IGNORE_UNUSED_ATTR.append(e)
