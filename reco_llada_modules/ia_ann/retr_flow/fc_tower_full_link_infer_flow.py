import os

from tower_public import *

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin 
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin


config_root_dir = os.path.join(os.path.dirname(__file__))
model_config = load_mio_tf_tower_model(
    config_root_dir,
    "fc_tower_full_link",
    "user_predict",
    "mio",
)

user_embedding_fetcher_config = dict( 
    user_feature_embedding_server="grpc_cascade_retrieval_distill_emb",
    user_feature_embedding_shard_num=16,
    user_embedding_request_type="predict_uemb_by_new_mc_tower",

)
user_embedding_predict_config = dict(
    dnn_model_queue_prefix="cascade_distill_exp_24_1126",
    receive_dnn_model_as_macro_block=True,
    rowmajor=True,
)

class MFlow(LeafFlow, MioApiMixin, OfflineApiMixin, EmbedCalcApiMixin):
    pass

fc_tower_full_link_infer_flow = (
    MFlow("fc_tower_full_link_infer_flow")
    .namespace_(ns="fc_tower_full_link_infer_flow", nest=True)
    .if_("enable_post_rank == 1")
    .if_("enable_post_rank_local_cache == 1")
        .local_cache_op(
            cache=[
                {"save": False, "common": True, "data_type": 0, "key_attr": "user_id", "value_attr": "user_top_layer", "cache_value_len": 544}, # 136 * 4
            ],
            memkv_expire_sec=300,
            memkv_capacity=30000000,
            memkv_mem_limit=214748364800,
        )
    .end_if_()
    .if_("user_top_layer == nil")
        .extract_with_ks_sign_feature(
            feature_list=model_config.feature_list,
            user_info_attr="user_info",
            common_slots_output="common_slots",
            common_parameters_output="common_parameters",
        )
        .fetch_mio_embedding(
            name="fetch_user_feature_embedding",
            kess_service=user_embedding_fetcher_config['user_feature_embedding_server'],
            protocol=user_embedding_fetcher_config.get('user_feature_embedding_protocal', 0),
            shards=user_embedding_fetcher_config['user_feature_embedding_shard_num'],
            timeout_ms=user_embedding_fetcher_config.get('user_feature_embedding_timeout_ms', 50),
            common_slots_inputs=["common_slots"],
            common_parameters_inputs=["common_parameters"],
            slots_config=model_config.slots_config,
            client_side_shard=user_embedding_fetcher_config.get('enable_client_side_shard', True),
            max_signs_per_request=user_embedding_fetcher_config.get('max_signs_num_per_request', 1000),
            thread_num=3,
            direct_write=True,
            save_result_as_tensor_output=True,
        )
        .mio_predict(
            graph=model_config.graph,
            queue_prefix=user_embedding_predict_config["dnn_model_queue_prefix"],
            key=user_embedding_predict_config["dnn_model_queue_prefix"],
            output_common_attr=True,
            inputs=[
                dict(
                    attr_name=c["input_name"],
                    tensor_name=c["input_name"],
                    common=c.get("common", False),
                    dim=len(str(c["slots"]).split(" ")) * c["dim"] * c.get("expand", 1)
                    + (1 if c.get("sized", False) else 0),
                )
                for c in model_config.slots_config
            ],
            outputs=[
                dict(
                    attr_name=attr_name,  # NOTE user top embedding 就在 attr_name 指向的 common_attr 里
                    tensor_name=tensor_name,
                )
                for attr_name, tensor_name in model_config.outputs
            ],
            flatten_outputs=True,
            param=model_config.param,
            receive_dnn_model_as_macro_block=user_embedding_predict_config.get('receive_dnn_model_as_macro_block',False),
            rowmajor=user_embedding_predict_config.get('rowmajor',False),  # kai 训练参数按行排，mio-tf按列排为 False
            read_input_from_extra_var=True,
        )
    .end_()
    .if_("enable_post_rank_local_cache == 1")        
        .local_cache_op(
            cache=[
                {"save": True, "common": True, "data_type":0, "key_attr":"user_id", "value_attr":"user_top_layer", "cache_value_len": 3536}, # 136 * 26
            ],
            memkv_expire_sec=300,
            memkv_capacity=30000000,
            memkv_mem_limit=214748364800,
        )
    .end_if_()    
    .gen_common_attr_by_lua(attr_map={
        "valid_user_id": "user_id > 0 and 1 or 0",
        "valid_user_info": "user_info ~= nil and 1 or 0",
        "valid_user_top_layer": "user_top_layer ~= nil and 1 or 0",
    })
    .if_("valid_user_top_layer ~= 1").return_(2, "invalid user top layer").end_()
    .end_if_()
    .perflog_attr_value(
        check_point="fc_tower_full_link_infer_flow.stats",
        common_attrs=["valid_user_id", "valid_user_info", "valid_user_top_layer"],
    )
    .namespace_()
)