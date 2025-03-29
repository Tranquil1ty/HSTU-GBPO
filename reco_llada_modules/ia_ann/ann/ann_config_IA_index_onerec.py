import sys

from ann_retrieve_flow import AnnRetrieveFlow
from dragonfly.common_leaf_dsl import LeafFlow, LeafService

kess = "grpc_ann_ia_128_index"
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

class GpuKnnRetrFlow(LeafFlow):
    def retrieve(self, emb_server_config = None, **kwargs):
        (self.enrich_attr_by_lua(
            import_common_attr=["photo_id_list"],
            export_common_attr=["photo_id_list_key"],
            function_for_common="func",
            lua_script="""
                function func()
                    local photo_id_list_key = {}
                    local semantic_id_str = ""
                    for i=1,#photo_id_list do
                        if i % 16 == 1 then
                            semantic_id_str = tostring(photo_id_list[i])
                        else
                            semantic_id_str = semantic_id_str ..'.'..tostring(photo_id_list[i])
                        end
                        if i % 16 == 0 then
                            table.insert(photo_id_list_key, util.CityHash64(semantic_id_str))
                        end
                    end
                return photo_id_list_key
                end
            """
        )
        .truncate(size_limit=0)
        .retrieve_by_local_ann(
            dest_bucket="photo_index",
            src_data_type="photo",
            src_items_attr="photo_id_list_key",
            src_embedding_list_attr="photo_emb_list",
            top_k="{{i2i_ann_topk}}",
            save_distance_to_attr="ann_score",
            save_src_item_to_attr="src_item",
            save_seq_num_to_attr="src_num",
            reason=1,
        )
        .copy_item_meta_info(save_item_id_to_attr="ann_pid")
        .log_debug_info(
            item_attrs=["src_num", "src_item", "ann_pid", "ann_score"],
            for_debug_request_only=True, 
            respect_sample_logging=False, 
        )
        .pack_item_attr(
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "aggregator": "concat",
                    "from_item_attr": "ann_pid",
                    "to_common_attr": "sim_photo_ids",
                },
                {
                    "aggregator": "concat",
                    # "from_item_attr": "filtered_ann_score",
                    "from_item_attr": "ann_score",
                    "to_common_attr": "sim_photo_scores",
                },
                {
                    "aggregator": "concat",
                    # "from_item_attr": "filtered_ann_score",
                    "from_item_attr": "src_item",
                    "to_common_attr": "sim_photo_src",
                },
                {
                    "aggregator": "concat",
                    # "from_item_attr": "filtered_ann_score",
                    "from_item_attr": "src_num",
                    "to_common_attr": "sim_photo_num",
                },
            ],
        )
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
service.return_common_attrs(attrs=["sim_photo_ids", "sim_photo_scores", "sim_photo_src", "sim_photo_num"])
# service.return_item_attrs(attrs=["ann_pid", "filtered_ann_score", "filtered_src_item"])
service.return_item_attrs(attrs=["ann_pid", "ann_score", "src_item"])
service.add_leaf_flows(request_type="cpu_knn", leaf_flows=[flow], as_default = True)

# 用于不需要输入 photo_emb_list 的 ANN demo
flow_demo = GpuKnnRetrFlow("cpu_knn_retr_demo").retrieve(emb_server_config = emb_server_config)
service.add_leaf_flows(request_type="cpu_knn_demo", leaf_flows=[flow_demo])

if __name__ == "__main__":
    service.build(
        output_file=__file__.replace(".py", ".json"),
    )
