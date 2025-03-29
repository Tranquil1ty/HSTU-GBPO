import sys

sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")

from ann_retrieve_flow import AnnRetrieveFlow
from dragonfly.common_leaf_dsl import LeafFlow, LeafService

kess = "grpc_AnnBigCodeEmd"
identifier = "i2i_bigcode_ann"
photo_embedding_btq = [f"new_ia_qwen2_{i}" for i in range(4)]
kconf_key = f"rinf.rlRunner.{identifier}"
SLOT = 16
EMBEDDING_DIM_BEGIN = 0
EMBEDDING_DIM_END= 896
TARGET_NUM = 1
TARGET_WEIGHTS = [1.0]
SINGLE_EMBEDDING_DIM = 896

# ANN 构建
datas = AnnRetrieveFlow().consume_data_from_btq(queue_names=photo_embedding_btq, thread_num=8) # 从 btq 消费数据
datas.configure_global_index(index_update_interval_second=60)
mio_data = datas.parse_data_in_mio(
    data_name="photo",
    max_item_num=10000000,
    dim=SINGLE_EMBEDDING_DIM * TARGET_NUM,
    kv_expire_second=3600*24*2, # 2 days
    slot_id=SLOT,
    parse_data_in_float=False,
    begin_bit=EMBEDDING_DIM_BEGIN,
    end_bit=EMBEDDING_DIM_END,
    use_plain_id_converter=True
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
    enable_auto_calc=False,
    enable_precision_eval=True,
    cache_expire_second=300, 
    cache_max_item_num=50000, 
)

class GpuKnnRetrFlow(LeafFlow):
    def retrieve(self, emb_server_config = None, **kwargs):
        self \
        .retrieve_by_common_attr(
            attr="photo_id_list",
            reason=1,
        ) \
        .copy_item_meta_info(
            save_item_id_to_attr="pid",
        ) 
        if emb_server_config is not None:
            # 从 btq 获取 photo embedding
            self.get_remote_embedding_lite_v2(
                kess_service = emb_server_config["kess_service"],
                shard_num = emb_server_config["shard_num"],
                id_converter=dict(type_name="plainIdConverter"),
                query_source_type = "item_attr",
                input_attr_name = "pid",
                output_attr_name = "photo_emb",
                client_side_shard = True,
                timeout_ms = 50,
                is_raw_data=False,
                size = emb_server_config["emb_size"],
            ) \
            .filter_by_attr(
                attr_name = "photo_emb",
                remove_if_attr_missing = True
            ) \
            .pack_item_attr(
                    item_source={
                        "reco_results": True,
                    },
                    mappings=[
                        {
                            "aggregator": "concat",
                            "from_item_attr": "photo_emb",
                            "to_common_attr": "photo_emb_list",
                        },
                    ],
                )
        self \
        .count_reco_result(
            save_count_to="photo_num",
        ) \
        .limit(0) \
        .retrieve_by_local_ann(
            dest_bucket="photo_index",
            src_data_type="photo",
            src_items_attr="photo_id_list",
            src_embedding_list_attr="photo_emb_list",
            top_k="{{i2i_ann_topk}}",
            save_distance_to_attr="ann_score",
            save_src_item_to_attr="src_item",
            reason=1,
        ) \
        .enrich_attr_by_lua(
            import_item_attr=["src_item", "ann_score"],
            export_item_attr=["filtered_src_item", "filtered_ann_score"],
            function_for_item="FilterLowScore",
            lua_script="""
            function FilterLowScore()
                -- 过滤低分的 ANN 召回
                local filtered_src_item = {}
                local filtered_ann_score = {}

                for i = 1, #ann_score do
                    if ann_score[i] >= 0.6 then
                        if src_item[i] ~= nil then
                            table.insert(filtered_ann_score, ann_score[i])
                            table.insert(filtered_src_item, src_item[i])
                        end
                    end
                end
                return filtered_src_item, filtered_ann_score
            end
            """,
        ) \
        .copy_item_meta_info(save_item_id_to_attr="ann_pid") \
        .filter_by_attr(
            attr_name="ann_pid",
            remove_if="<=",
            compare_to=0,
            remove_if_attr_missing=True,
        ) \
        .pack_item_attr(
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "aggregator": "concat",
                    "from_item_attr": "filtered_src_item",
                    "to_common_attr": "sim_photo_ids",
                },
                {
                    "aggregator": "concat",
                    "from_item_attr": "filtered_ann_score",
                    "to_common_attr": "sim_photo_scores",
                },
            ],
        ) \
        .gen_common_attr_by_lua(
            attr_map={
                "sim_photo_id_num": "#(sim_photo_ids or {})",
                "sim_photo_score_num": "#(sim_photo_scores or {})",
            }
        ) \
        .perflog_attr_value(
            check_point=f"{identifier}.result",
            common_attrs=[
                "sim_photo_id_num",
                "sim_photo_score_num",
                "i2i_ann_topk",
            ],
        )
        return self


flow = GpuKnnRetrFlow("cpu_knn_retr").retrieve()
service = LeafService(
    kess_name=kess,
    common_attrs_from_request=["photo_id_list", "photo_emb_list", "i2i_ann_topk"],
    ann_config=knn_retr.get_config(),
)

service.IGNORE_NO_SOURCE_ATTR = ["item_key"]
LeafService.CHECK_UNUSED_ATTR = False
service.return_common_attrs(attrs=["sim_photo_ids", "sim_photo_scores"])
service.add_leaf_flows(request_type="cpu_knn", leaf_flows=[flow], as_default = True)

emb_server_config = {
  "kess_service": "ia-qwen2-emb",
  "shard_num": 32,
  "emb_size": 896
}
# 用于不需要输入 photo_emb_list 的 ANN demo
flow_demo = GpuKnnRetrFlow("cpu_knn_retr_demo").retrieve(emb_server_config = emb_server_config)
service.add_leaf_flows(request_type="cpu_knn_demo", leaf_flows=[flow_demo])

if __name__ == "__main__":
    service.build(
        output_file=__file__.replace(".py", ".json"),
    )
