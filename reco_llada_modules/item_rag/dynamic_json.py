#!/usr/bin/env python3
# coding=utf-8

import os, sys
import base64
import collections
import yaml


from dragonfly.common_leaf_dsl import LeafService, IndexSource, LeafFlow
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.uni_predict_v2.uni_predict_v2_api_mixin import UniPredictV2ApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin

from ann_retrieve_flow import AnnRetrieveFlow
from extract_user_his import user_seq_flow, returned_user_seq_attrs

service_name= "grpc_item_rag_server"

item_RAG_num = 32
token_num = 16
item_rag_attrs = ["1520", "1606", "1607", "26", "128", "71", "93", "141", "142", "143", "417", "418", "430", "776", "777", "778", "779", "780", "781", "782"]
returned_item_rag_attrs = [str(int(attr) + 30000) for attr in item_rag_attrs]

############## ANN ################

identifier = "i2i_ann_ia_128_index_config"
photo_embedding_btq = [f"ia_emb_128_for_ann{i}" for i in range(4)]
kconf_key = f"rinf.rlRunner.{identifier}"

SLOT = 17
SINGLE_EMBEDDING_DIM = 128
EMBEDDING_DIM_BEGIN = 0
EMBEDDING_DIM_END =  SINGLE_EMBEDDING_DIM
TARGET_NUM = 1
TARGET_WEIGHTS = [1.0]


emb_server_config = {
  "colossusdb_embd_model_name": "wxm_mm_sim_gsu_emb",
  "colossusdb_embd_table_name": "emb_wxm_mm_sim_gsu_128_new",
  "shard_num": 16,
  "emb_size": SINGLE_EMBEDDING_DIM, 
}

# ANN 构建
datas = AnnRetrieveFlow().consume_data_from_btq(queue_names=photo_embedding_btq, thread_num=8) # 从 btq 消费数据
datas.configure_global_index(index_update_interval_second=60)
mio_data = datas.parse_data_in_mio(
    data_name="photo",
    max_item_num=10000000,
    dim=SINGLE_EMBEDDING_DIM * TARGET_NUM,
    kv_expire_second=3600*6, # 2 days
    slot_id=SLOT,
    parse_data_in_float=False,
    begin_bit=EMBEDDING_DIM_BEGIN,
    end_bit=EMBEDDING_DIM_END,
    use_plain_id_converter=False,
    # id_converter="plainIdConverter",
)
photo_index = mio_data.build_scann_index(
    bucket_name="photo_index",
    space="ip", 
    final_neighbors_num=1000,
    leaves_num=1100,
    leaves_to_search=55,
    pre_reorder_neighbors_num=2400,
    training_sample_size=300000,
    anisotropic_quantization_threshold=0.4,
    threshold_kconf_key=kconf_key,
)

knn_retr = mio_data.retrieve_from(
    dest_bucket=photo_index,
    enable_auto_calc=True,
    enable_precision_eval=True,
    cache_expire_second=600, 
    cache_max_item_num=50000, 
)

############## ANN ################

############## Decoder ################
queue_prefix = "semantic_id_decoder"   #

model_key = "semantic_id_decoder"

ModelConfig = collections.namedtuple(
    "ModelConfig",
    [
        "graph",
        "outputs",
        "param",
    ],
)

def load_tf_model():
    model_dir = os.path.join(os.path.dirname(__file__))

    with open(os.path.join(model_dir, 'dnn_model.yaml')) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(model_dir, 'graph.pb'), 'rb') as f:
        base64_graph = base64.b64encode(f.read()).decode('ascii')
        graph = 'base64://' + base64_graph

    q_names = dnn_model['q_names'].split(' ')

    graph_tensor_mapping = dnn_model['graph_tensor_mapping']
    outputs = [(x, graph_tensor_mapping[x]) for x in q_names]
    param = [param for param in dnn_model['param'] if param.get('send_to_online', True)]

    return ModelConfig(
        graph,
        outputs,
        param
    )

model_config = load_tf_model()
############## Decoder ################

class PrepareFlow(LeafFlow):

    def prepare(self):
        
        kconf_configs_var = {
            "semantic_id_decoder_kess": "grpc_semantic_id_decoder_cpu",
            "decoder_timeout_ms": 50,
            "rag_ann_topk": item_RAG_num,
            "ann_timeout_ms": 50,
            "ann_kess": "grpc_ann_ia_128_index_offline",
            "vocab_size": 512,
            "use_item_rag": 1,
            "use_user_seq": 1,
            "use_user_rag": 1,
            "target_as_result": 0
        }

        kconf = [
            {"kconf_key": "rinf.rlRunner.rag_config", "export_common_attr": k, "json_path": k, "default_value": v}
            for k, v in kconf_configs_var.items()
        ]

        return (
            self.get_kconf_params(kconf_configs=kconf)
                .copy_user_meta_info(save_request_type_to_attr="request_type")
                .copy_item_meta_info(save_item_key_to_attr="photo_id")
        )

class DecodeFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, EmbedCalcApiMixin, GsuApiMixin, PDNApiMixin, CofeaApiMixin, UniPredictV2ApiMixin, EmbeddingApiMixin):

    def train_mask_tokens(self):
        return (
            self.enrich_attr_by_lua(
                import_common_attr = ["vocab_size"],
                import_item_attr = ["tokens"],
                # function_for_item 的值也可用 "{{}}" 格式指定为某个 common_attr
                function_for_item = "calculate",
                # 将 calculate 函数的返回值依次存入 export_item_attr 指定的 3 个 item_attr 中
                export_item_attr = ["tokens"],
                lua_script = """
                    function calculate(seq, item_key, reason, score)
                        local mask_token = vocab_size or 512
                        local tokens = tokens
                        local t = util.Random()
                        local masked_tokens = {}
                        sum = 0
                        for i=1, #tokens do
                            p = util.Random()
                            if p < t then
                                table.insert(masked_tokens, mask_token)
                            else
                                table.insert(masked_tokens, tokens[i])
                            end
                            sum = sum + tokens[i]
                        end
                        if sum == 0 then
                            return tokens
                        else
                            return masked_tokens
                        end
                    end
                """
            )
        )

    def decode(self, **kwargs):
        return (
            self
            .enrich_attr_by_lua(
                import_item_attr = ["semantic_id_v2", "tokens"],
                function_for_item = "calculate",
                export_item_attr = ["tokens"],
                lua_script = """
                    function calculate(seq, item_key, reason, score)
                        if semantic_id_v2 ~= nil then
                            return semantic_id_v2
                        else
                            return tokens
                        end
                    end
                """
            )
            .set_attr_value(
                no_overwrite=True,
                item_attrs=[
                    {"name": "tokens", "type": "int_list", "value": [0] * 16}
                ]
            )
            .if_("request_type == 'train_request'")
                .train_mask_tokens()
            .end_if_()
            .cast_attr_type(
                attr_type_cast_configs=[
                    {
                    "to_type": "double",
                    "from_item_attr": "tokens",
                    "to_item_attr": "tokens_double"
                    },
                ]
            )       
            .uni_predict_fused(
                static_graph=True,
                graph=model_config.graph,
                key=model_key,
                queue_prefix=queue_prefix,
                inputs=[
                    {
                        "attr_name": "tokens_double",
                        "tensor_name": "tokens",
                        "dim": 16
                    }
                ],
                outputs=[dict(attr_name=attr_name, tensor_name=tensor_name) for attr_name, tensor_name in model_config.outputs],
                param=model_config.param,
                model_loader_config=dict(#rowmajor=True,
                                        type="MioTFExecutedByTensorFlowModelLoader",
                                        # implicit_batch=False,
                                        dynamic_shape=True,
                                        executor_batchsizes=[256],  # user 单边预估，batch_size 为 1
                                        receive_dnn_model_as_macro_block=False,
                                        enable_xla=False,
                                        enable_bf16=False,
                                        force_input_tensor_fp32=True,
                                        # enable_fp16=False,
                                        ),
                batching_config=dict(batch_timeout_micros=0,
                                    max_batch_size=256,
                                    max_enqueued_batches=1,
                                    batch_task_type="BatchTensorflowTask"),
                executor_config=dict(intra_op_parallelism_threads_num=32,
                                    inter_op_parallelism_threads_num=32,
                                    context_per_device=0,
                                    )
            )
            .log_debug_info(
                item_attrs = ["embs"], 
                for_debug_request_only=True, 
                respect_sample_logging=False
            )
        )
    

class ItemRAGFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, EmbedCalcApiMixin, GsuApiMixin, PDNApiMixin, CofeaApiMixin, UniPredictV2ApiMixin, EmbeddingApiMixin):

    def run_item_rag(self):
        return (
            self.if_("use_item_rag == 1")
            .ann_retrieve()
            .get_item_attributes()
            .extract_features()
            .end_if_()
        )

    def run_item_rag_infer(self):
        return (
            self.if_("use_item_rag == 1")
            .ann_retrieve()
            .get_item_attributes()
            .extract_features()
            .infer_format()
            .end_if_()
        )

    def ann_retrieve(self):
        return (
            self
            .if_("request_type == 'train_request' and target_as_result == 1")
                .set_attr_value(
                    no_overwrite=True,
                    item_attrs=[
                        {
                            "name": "ann_score",
                            "type": "double",
                            "value": 1.1
                        }
                    ]
                )
                .pack_item_attr(
                    item_source = {
                        "reco_results": True
                    },
                    mappings = [
                        {"to_common_attr": "req_keys"},
                        {"from_item_attr": "photo_id", "to_common_attr": "ann_pids"},
                        {"from_item_attr": "ann_score", "to_common_attr": "ann_scores"}
                    ]
                )
            .else_()
                .pack_item_attr(
                    item_source = {
                        "reco_results": True
                    },
                    mappings = [
                        {"to_common_attr": "req_keys"},
                        {"from_item_attr": "embs", "to_common_attr": "embs_list"},
                        {"from_item_attr": "tokens", "to_common_attr": "tokens_list"},
                    ]
                )
                .log_debug_info(
                    common_attrs = ["req_keys"], 
                    for_debug_request_only=True, 
                    respect_sample_logging=False
                )
                .count_reco_result(save_count_to="item_num")
                .enrich_attr_by_lua(
                    import_common_attr = ["rag_ann_topk", "ann_topk"],
                    function_for_common = "calculate",
                    export_common_attr = ["i2i_ann_topk"],
                    lua_script = """
                        function calculate()
                            if ann_topk ~= nil then
                                return ann_topk
                            else
                                return rag_ann_topk
                            end
                        end
                    """
                )
                .gen_common_attr_by_lua(attr_map={"ann_request_num": "i2i_ann_topk * item_num"})
                .truncate(size_limit=0)
                .delegate_retrieve(
                    kess_service="{{ann_kess}}",
                    timeout_ms="{{ann_timeout_ms}}",
                    send_common_attrs=[
                        "tokens_list",
                        "embs_list",
                        "i2i_ann_topk",
                    ],
                    recv_item_attrs=[
                        "ann_score", "ann_pid"
                    ],
                    request_type="cpu_knn",
                    request_num="{{ann_request_num}}",
                    reason=3
                )
                .log_debug_info(
                    item_attrs=["ann_pid", "ann_score"],
                    for_debug_request_only=True, 
                    respect_sample_logging=False, 
                )
                .pack_item_attr(
                    item_source = {
                        "reco_results": True
                    },
                    mappings = [
                        {"from_item_attr": "ann_pid", "to_common_attr": "ann_pids"},
                        {"from_item_attr": "ann_score", "to_common_attr": "ann_scores"}
                    ]
                )
            .end_if_()

        )

    def load_feature_list_sign(self, filename):
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

    def get_item_attributes(self):
        return (
            self.get_item_attr_by_distributed_new_photo_info_index(
                photo_store_kconf_key = "reco.model2.recoExploreFastPhotoStoreConfigForOnerec",
                save_item_info_to_attr="photo_info"
            )
            .enrich_with_protobuf(
                from_extra_var="photo_info",
                is_common_attr=False,
                attrs=[
                    "photo_id",
                    "user_hash_tag_id",
                    "duration_ms",
                    dict(path="author.id", name="author_id"),
                    dict(path="search_query_id", name="search_bubble_query_id"),
                    dict(path="explore_stat.real_show_count", name="realshow_count"),
                    dict(path="explore_stat.like_count", name="like_count"),
                    dict(path="explore_stat.follow_count", name="follow_count"),
                    dict(path="explore_stat.forward_count", name="forward_count"),
                    dict(path="explore_stat.profile_enter_count", name="profile_enter_count"),
                    dict(path="explore_stat.negative_count", name="negative_count"),
                    dict(path="explore_stat.comment_count", name="comment_count"),
                    dict(path="explore_stat.short_play_count", name="short_play_count"),
                    dict(path="explore_stat.full_play_count", name="full_play_count"),
                    dict(path="location.lat", name="lat"),
                    dict(path="location.lon", name="lon"),
                    "upload_time",
                    dict(path="local_life_photo_info.poi_info.new_poi_id", name="local_life_new_poi_id"),
                    dict(path="location.poi_city_id", name="local_life_poi_city_id"),
                    "reco_playlet_tag",
                    dict(path="video_playlet_info.name_hash", name="playlet_name"),
                    dict(path="video_playlet_info.theme_hash", name="playlet_theme"),
                    dict(path="video_playlet_info.channel_hash", name="playlet_channel"),
                    dict(path="video_playlet_info.plot_hash", name="playlet_plot"),
                ]
            )
        )

    def extract_features(self):

        feature_list = self.load_feature_list_sign('feature_list_sign.txt')
        
        return (
            self.enrich_attr_by_lua(
                import_common_attr=['_REQ_TIME_'],
                import_item_attr=["upload_time"],
                export_item_attr=["photo_age_hour"],
                function_for_item="calculate",
                lua_script="""
                    function calculate()
                        local hour_ms = 60*60*1000
                        local photo_age_hour = 0
                        if upload_time ~= nil and _REQ_TIME_ ~= nil and (_REQ_TIME_ - upload_time) >= 0 then
                        local photo_age_ms = _REQ_TIME_ - upload_time
                        photo_age_hour = math.floor(photo_age_ms / hour_ms)
                        end
                        return photo_age_hour
                    end
                """
            )
            .extract_with_ks_sign_feature(
                feature_list=feature_list,
                photo_info_attr="photo_info",
                slot_as_attr_name=True
            )
            .extract_kuiba_parameter(
                config={
                    "photo_age_hour": {"attrs": [{"key_type": 1520, "attr": ["photo_age_hour"], "converter": "discrete", "converter_args": "3000,0.00,1,100,0.2"}]},
                    'local_life_poi_city_id': {'attrs': [{'key_type': 1606, "mio_slot_key_type": 1606, 'attr': ['local_life_poi_city_id'], "converter": "id"}]},
                    'local_life_new_poi_id': {'attrs': [{'key_type': 1607, "mio_slot_key_type": 1607, 'attr': ['local_life_new_poi_id'], "converter": "id"}]},
                }, 
                is_common_attr=False,
                slot_as_attr_name=True
            )
            .set_attr_value(
                no_overwrite=True,
                item_attrs=[
                    {"name": attr, "type": "int_list", "value": [((int(attr)-16) << 48) | ((1<<48)-1)] * (2 if attr == "93" else 3 if attr == "418" else 1)} for attr in item_rag_attrs
                ]
            )
            .log_debug_info(
                item_attrs = ["1520", "1606", "1607", "26", "128", "71", "93", "141", "142", "143", "417", "418", "430", "776", "777", "778", "779", "780", "781", "782"], 
                for_debug_request_only=True, 
                respect_sample_logging=False, 
            )
            .pack_item_attr(
                item_source = {
                    "reco_results": True
                },
                mappings = [
                    {"from_item_attr": attr, "to_common_attr": remap_attr} for attr, remap_attr in zip(item_rag_attrs, returned_item_rag_attrs)
                ]
            )
            .if_("request_type == 'train_request' and target_as_result == 1")
                .dispatch_common_attr(
                    dispatch_config = [
                        {"from_common_attr" : "ann_pids", "to_item_attr" : "ann_pids", "by_list_size": 1},
                        {"from_common_attr" : "ann_scores", "to_item_attr" : "ann_scores", "by_list_size": 1}
                    ] + [
                        {"from_common_attr" : attr, "to_item_attr" : attr, "by_list_size": 1 *2 if attr == "93" else 1 * 3 if attr=="418" else 1} for attr in returned_item_rag_attrs
                    ]
                )
            .else_()
                .truncate(size_limit=0)
                .retrieve_by_common_attr(attr="req_keys", reason=666)
                .copy_item_meta_info(save_item_key_to_attr="photo_id")
                .dispatch_common_attr(
                    dispatch_config = [
                        {"from_common_attr" : "embs_list", "to_item_attr" : "embs", "by_list_size": SINGLE_EMBEDDING_DIM},
                        {"from_common_attr" : "tokens_list", "to_item_attr" : "tokens", "by_list_size": token_num},
                        {"from_common_attr" : "ann_pids", "to_item_attr" : "ann_pids", "by_list_size": item_RAG_num},
                        {"from_common_attr" : "ann_scores", "to_item_attr" : "ann_scores", "by_list_size": item_RAG_num}
                    ] + [
                        {"from_common_attr" : attr, "to_item_attr" : attr, "by_list_size": item_RAG_num *2 if attr == "93" else item_RAG_num * 3 if attr=="418" else item_RAG_num} for attr in returned_item_rag_attrs
                    ]
                )
            .end_if_()
            .enrich_attr_by_lua(
                import_item_attr = ["photo_id", "ann_pids"],
                function_for_item = "calculate",
                export_item_attr = ["target_inside"],
                lua_script = """
                    function calculate(seq, item_key, reason, score)
                        for i=1, #ann_pids do
                            if photo_id == ann_pids[i] then
                                return 1
                            end
                        end
                        return 0
                    end
                """
            )
            .log_debug_info(
                item_attrs = ["target_inside"], 
                for_debug_request_only=True, 
                respect_sample_logging=False, 
            )
            .perflog_attr_value(
                check_point="{{return 'item_rag.target_inside.' .. request_type}}",
                item_attrs=["target_inside"],
            )
        )

    def infer_format(self):
        return (
            self.pack_item_attr_to_item_attr(
                from_item_attrs=returned_item_rag_attrs,
                to_item_attr="item_rag_parameters",
                default_val=[0]
            )
            .enrich_attr_by_lua(
                import_item_attr=["item_rag_parameters"],
                export_item_attr=["item_rag_slots"],
                function_for_item="calculate",
                lua_script="""
                    function calculate()
                        item_rag_slots = {}
                        for i=1, #item_rag_parameters do
                            slot = (item_rag_parameters[i] >> 48) + 16 + 30000
                            table.insert(item_rag_slots, slot)
                        end
                        return item_rag_slots
                    end
                """
            )
        )

prepare_flow = PrepareFlow(name="prepare_flow").prepare()

decode_flow = DecodeFlow(name="decode_flow").decode()

ann_flow = ItemRAGFlow(name="ann_flow").ann_retrieve()
item_rag_flow = ItemRAGFlow(name="item_rag_flow").run_item_rag()
item_rag_infer_flow = ItemRAGFlow(name="item_rag_infer_flow").run_item_rag_infer()

kess_name = service_name
print(f"kess name: {kess_name}")

service = LeafService(
    kess_name=kess_name,
    item_attrs_from_request=["tokens", "semantic_id_v2", "time_ms"],
    common_attrs_from_request=["user_seq_size", "ann_topk"],
    index_source=IndexSource.LOCAL_ATTR_INDEX,
    ann_config=knn_retr.get_config(),
)

service.AUTO_INJECT_ITEM_ATTR = False
service.CHECK_UNUSED_ATTR = False
# service.CHECK_NO_SOURCE_ATTR = False

service.IGNORE_NO_SOURCE_ATTR=returned_user_seq_attrs
returned_item_attrs = ["tokens", "embs", "item_rag_slots", "item_rag_parameters", "ann_pid", "ann_score", "ann_pids", "ann_scores"] + returned_item_rag_attrs + returned_user_seq_attrs
service.return_item_attrs(attrs=returned_item_attrs)
returned_common_attrs = ["tokens", "ann_pids", "ann_scores", "embs_list", "user_seq_slots", "user_seq_parameters", "colossus_time_s"] + returned_item_rag_attrs
service.return_common_attrs(attrs=returned_common_attrs)

service.add_leaf_flows(request_type="default", leaf_flows=[prepare_flow, decode_flow, item_rag_flow, ], as_default=True)
service.add_leaf_flows(request_type="item_rag_request", leaf_flows=[prepare_flow, decode_flow, item_rag_flow, ])
service.add_leaf_flows(request_type="user_seq_request", leaf_flows=[prepare_flow, user_seq_flow, ])
service.add_leaf_flows(request_type="decode_request", leaf_flows=[prepare_flow, decode_flow, ])
service.add_leaf_flows(request_type="ann_request", leaf_flows=[prepare_flow, decode_flow, ann_flow])
service.add_leaf_flows(request_type="train_request", leaf_flows=[prepare_flow, decode_flow, item_rag_flow, user_seq_flow])
service.add_leaf_flows(request_type="infer_request", leaf_flows=[prepare_flow, decode_flow, item_rag_infer_flow, user_seq_flow])

service.build(output_file=__file__.replace(".py", ".json"))

'''
kuiba_parameter_non_common_config = {
  "pid": {"attrs": [{"key_type": 736, "attr": ["photo_id"], "converter": "id"}]},
  "aid": {"attrs": [{"key_type": 737, "attr": ["author_id"], "converter": "id"}]},
#   "cluster_id_fea": {"attrs": [{"key_type": 800, "attr": ["target_photo_cluster_id"], "converter": "id"}]},
#   "user_hash_tag_id" : {"attrs" : [{"key_type": 727, "attr": ["user_hash_tag_id"], **kuiba_list_converter_config}]},
#   "wtd_bucket" : {"attrs" : [{"key_type": 1083, "attr": ["wtd_bucket"], "converter": "id"}]},
#   "wtd_bucket_duration" : {"attrs" : [{"key_type": 1084, "attr": ["wtd_bucket_duration"], "converter": "id"}]},
#   "wtd_evtr_bucket" : {"attrs" : [{"key_type": 1085, "attr": ["wtd_bucket"], "converter": "id"}]},
#   "wtd_evtr_bucket_duration" : {"attrs" : [{"key_type": 1086, "attr": ["wtd_bucket_duration"], "converter": "id"}]},
#   "wtd_lvtr_bucket" : {"attrs" : [{"key_type": 1087, "attr": ["wtd_bucket"], "converter": "id"}]},
#   "wtd_lvtr_bucket_duration" : {"attrs" : [{"key_type": 1088, "attr": ["wtd_bucket_duration"], "converter": "id"}]},
#   "search_top4_pid": {"attrs": [{"mio_slot_key_type": 1313, "key_type": 26, "attr": ["search_bubble_top4_pid"], **kuiba_list_converter_config}]},
#   "search_bubble_query_ctr_param": {"attrs": [{"key_type": 1314, "attr": ["search_bubble_query_ctr_bucket"], "converter": "id"}]},
#   "search_bubble_query_score_param": {"attrs": [{"key_type": 1315, "attr": ["search_bubble_query_score_bucket"], "converter": "id"}]},
#   "search_bubble_query_show_param": {"attrs": [{"key_type": 1316, "attr": ["search_bubble_query_show_count_bucket"], "converter": "id"}]},
#   "search_bubble_query_click_param": {"attrs": [{"key_type": 1317, "attr": ["search_bubble_query_click_count_bucket"], "converter": "id"}]},\
#   "photo_geohash_2": {"attrs": [{"key_type": 1512, "attr": ["lat", "lon"], "converter": "geohash", "converter_args": "2"}]},
#   "photo_age_hour": {"attrs": [{"key_type": 1520, "attr": ["photo_age_hour"], "converter": "discrete", "converter_args": "3000,0.00,1,100,0.2"}]},
  'local_life_poi_city_id': {'attrs': [{'key_type': 1606, "mio_slot_key_type": 1606, 'attr': ['local_life_poi_city_id'], "converter": "id"}]},
  'local_life_new_poi_id': {'attrs': [{'key_type': 1607, "mio_slot_key_type": 1607, 'attr': ['local_life_new_poi_id'], "converter": "id"}]},
#   "query_search_click_1d": {"attrs": [{"key_type": 2010, "attr": ["q_s_click_1d"], "converter": "id"}]},
#   "query_search_result_click_PV_1d": {"attrs": [{"key_type": 2011, "attr": ["q_s_r_click_PV_1d"], "converter": "id"}]},
#   "query_search_result_play_PV_1d": {"attrs": [{"key_type": 2012, "attr": ["q_s_r_play_PV_1d"], "converter": "id"}]},
#   "query_search_result_time_1d": {"attrs": [{"key_type": 2013, "attr": ["q_s_r_time_1d"], "converter": "id"}]},
#   "query_search_result_imp_item_1d": {"attrs": [{"key_type": 2014, "attr": ["q_s_r_imp_item_1d"], "converter": "id"}]},
#   "query_search_result_click_item_1d": {"attrs": [{"key_type": 2015, "attr": ["q_s_r_click_item_1d"], "converter": "id"}]},
#   "query_search_result_play_item_1d": {"attrs": [{"key_type": 2016, "attr": ["q_s_r_play_item_1d"], "converter": "id"}]},
#   "query_search_click_7d": {"attrs": [{"key_type": 2017, "attr": ["q_s_click_7d"], "converter": "id"}]},
#   "query_search_result_click_PV_7d": {"attrs": [{"key_type": 2018, "attr": ["q_s_r_click_PV_7d"], "converter": "id"}]},
#   "query_search_result_play_PV_7d": {"attrs": [{"key_type": 2019, "attr": ["q_s_r_play_PV_7d"], "converter": "id"}]},
#   "query_search_result_time_7d": {"attrs": [{"key_type": 2020, "attr": ["q_s_r_time_7d"], "converter": "id"}]},
#   "query_search_result_imp_item_7d": {"attrs": [{"key_type": 2021, "attr": ["q_s_r_imp_item_7d"], "converter": "id"}]},
#   "query_search_result_click_item_7d": {"attrs": [{"key_type": 2022, "attr": ["q_s_r_click_item_7d"], "converter": "id"}]},
#   "query_search_result_play_item_7d": {"attrs": [{"key_type": 2023, "attr": ["q_s_r_play_item_7d"], "converter": "id"}]},
#   "query_search_comment_click_1d": {"attrs": [{"key_type": 2024, "attr": ["q_s_click_1d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_click_PV_1d": {"attrs": [{"key_type": 2025, "attr": ["q_s_r_click_PV_1d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_play_PV_1d": {"attrs": [{"key_type": 2026, "attr": ["q_s_r_play_PV_1d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_time_1d": {"attrs": [{"key_type": 2027, "attr": ["q_s_r_time_1d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_imp_item_1d": {"attrs": [{"key_type": 2028, "attr": ["q_s_r_imp_item_1d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_click_item_1d": {"attrs": [{"key_type": 2029, "attr": ["q_s_r_click_item_1d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_play_item_1d": {"attrs": [{"key_type": 2030, "attr": ["q_s_r_play_item_1d_cmt"], "converter": "id"}]},
#   "query_search_comment_click_7d": {"attrs": [{"key_type": 2031, "attr": ["q_s_click_7d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_click_PV_7d": {"attrs": [{"key_type": 2032, "attr": ["q_s_r_click_PV_7d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_play_PV_7d": {"attrs": [{"key_type": 2033, "attr": ["q_s_r_play_PV_7d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_time_7d": {"attrs": [{"key_type": 2034, "attr": ["q_s_r_time_7d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_imp_item_7d": {"attrs": [{"key_type": 2035, "attr": ["q_s_r_imp_item_7d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_click_item_7d": {"attrs": [{"key_type": 2036, "attr": ["q_s_r_click_item_7d_cmt"], "converter": "id"}]},
#   "query_search_comment_result_play_item_7d": {"attrs": [{"key_type": 2037, "attr": ["q_s_r_play_item_7d_cmt"], "converter": "id"}]},
#   "search_queryterm": {"attrs": [{"key_type": 2000, "attr": ["search_bubble_query_term"], **kuiba_list_converter_config}]},
#   "search_querydura": {"attrs": [{"key_type": 2001, "attr": ["search_bubble_query_term_time"], **kuiba_list_converter_config}]},
  "fuse_duration_fea": {"attrs": [{"key_type": 1394, "converter": "id", "attr": ["duration"]}]},
#   "playlet_name_out": {"attrs": [{"key_type": 2201, "attr": ["playlet_name"], "converter": "id"}]},
#   "playlet_theme_out": {"attrs": [{"key_type": 2202, "attr": ["playlet_theme"], "converter": "id"}]},
#   "playlet_channel_out": {"attrs": [{"key_type": 2203, "attr": ["playlet_channel"], "converter": "id"}]},
#   "playlet_plot_out": {"attrs": [{"key_type": 2204, "attr": ["playlet_plot"], "converter": "id"}]},
}
'''