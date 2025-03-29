import sys

sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")

from ann_retrieve_flow import AnnRetrieveFlow
from dragonfly.common_leaf_dsl import LeafFlow, LeafService

kess = "grpc_AnnBigCodeIndexEmd"
identifier = "i2i_bigcode_ann"
photo_embedding_btq = ["bigcode_index_emb0"]
kconf_key = f"rinf.rlRunner.{identifier}"

EMBEDDING_DIM_BEGIN = 0
EMBEDDING_DIM_END= 896
TARGET_NUM = 1
TARGET_WEIGHTS = [1.0]
SINGLE_EMBEDDING_DIM = 896

# ANN 构建
datas = AnnRetrieveFlow().consume_data_from_btq(queue_names=photo_embedding_btq, thread_num=8) # 从 btq 消费数据
datas.configure_global_index(index_update_interval_second=60)

# 从runner得到的索引池中的高质量视频embedding（slot=17），构建ann
mio_data_from_index = datas.parse_data_in_mio(
    data_name="photo",
    max_item_num=10000000,
    dim=SINGLE_EMBEDDING_DIM * TARGET_NUM,
    kv_expire_second=3600*24*2, # 2 days
    slot_id=17,
    parse_data_in_float=False,
    begin_bit=EMBEDDING_DIM_BEGIN,
    end_bit=EMBEDDING_DIM_END,
    use_plain_id_converter=False
)
photo_index_from_index = mio_data_from_index.build_scann_index(
    bucket_name="photo_index_from_index",
    space="ip", 
    final_neighbors_num=1000,
    leaves_num=1100,
    leaves_to_search=100,
    pre_reorder_neighbors_num=5000,
    training_sample_size=300000,
    anisotropic_quantization_threshold=0.4,
    threshold_kconf_key=kconf_key,
)

knn_retr_from_index = mio_data_from_index.retrieve_from(
    dest_bucket=photo_index_from_index,
    enable_auto_calc=True,
    enable_precision_eval=True,
    cache_expire_second=3600, 
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
            dest_bucket="photo_index_from_index",
            src_data_type="photo",
            src_items_attr="photo_id_list",
            src_embedding_list_attr="photo_emb_list",
            top_k="{{i2i_ann_topk}}",
            save_distance_to_attr="ann_score",
            save_src_item_to_attr="src_item",
            reason=1,
        ) \
        .copy_item_meta_info(save_item_id_to_attr="ann_pid") \
        .filter_by_attr(
            attr_name="ann_pid",
            remove_if="<=",
            compare_to=0,
            remove_if_attr_missing=True,
        ) 
        # .get_local_ann_embedding(
        #     src_data_type = "photo",
        #     dim = SINGLE_EMBEDDING_DIM,
        #     save_to_common_attr = False,
        #     embedding_item_attr = 'emb_from_ann'
        # ) \
        self.enrich_attr_by_lua(
            import_item_attr=["src_item", "ann_score"],
            export_item_attr=["filtered_src_item", "filtered_ann_score", "low_ann_score"],
            function_for_item="FilterLowScore",
            lua_script="""
            function FilterLowScore()
                -- 过滤低分的 ANN 召回
                local filtered_src_item = {}
                local filtered_ann_score = {}
                local low_ann_score = 0
                for i = 1, #ann_score do
                    if ann_score[i] >= 0.8 then
                        if src_item[i] ~= nil then
                            table.insert(filtered_ann_score, ann_score[i])
                            table.insert(filtered_src_item, src_item[i])
                        end
                    else
                        low_ann_score = 1
                    end   
                end
                return filtered_src_item, filtered_ann_score, low_ann_score
            end
            """,
        ) \
        .filter_by_attr(
            attr_name="low_ann_score",
            remove_if="==",
            compare_to=1,
            remove_if_attr_missing=True,
        ) 
        return self


flow = GpuKnnRetrFlow("cpu_knn_retr").retrieve()
service = LeafService(
    kess_name=kess,
    common_attrs_from_request=["photo_id_list", "photo_emb_list", "i2i_ann_topk"],
    ann_config=knn_retr_from_index.get_config(),
)

service.IGNORE_NO_SOURCE_ATTR = ["item_key"]
LeafService.CHECK_UNUSED_ATTR = False
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
