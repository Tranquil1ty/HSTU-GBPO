import sys

sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")

from ann_retrieve_flow import AnnRetrieveFlow
from dragonfly.common_leaf_dsl import LeafFlow, LeafService

kess = "grpc_AnnIABase",
identifier = "mmu_e2e_i2i_index_ann"
kconf_key = f"rinf.rlRunner.{identifier}_config"
# photo_embedding_btq = [f"mmu_e2e_i2i_emb{i}" for i in range(32)]
photo_embedding_btq = ["mmu_i2i_index_emb0"]
SLOT = 16
EMBEDDING_DIM_BEGIN = 0
EMBEDDING_DIM_END= 512
TARGET_NUM = 1
TARGET_WEIGHTS = [1.0]
SINGLE_EMBEDDING_DIM = 512 

# 从 btq 消费数据
datas = AnnRetrieveFlow().consume_data_from_btq(queue_names=photo_embedding_btq, thread_num=4)
datas.configure_global_index(index_update_interval_second=60)
mio_data = datas.parse_data_in_mio(
    data_name="photo",
    max_item_num=10000000,
    dim=SINGLE_EMBEDDING_DIM * TARGET_NUM,
    kv_expire_second=3600*6,
    slot_id=SLOT,
    parse_data_in_float=True,
    begin_bit=EMBEDDING_DIM_BEGIN,
    end_bit=EMBEDDING_DIM_END,
    use_plain_id_converter=True,
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
) \
.filter_index_by_lua_script(
    import_item_attr=[
        "photo_id", "duration_ms", 
        "upload_time", "long_term_photo",
        "explore_stat__real_show_count", 
        "thanos_stats__real_show_count", 
        "nebula_stats__real_show_count"
    ],
    item_remove_check_func="item_should_remove",
    remove_if_lua_fail=True,
    lua_script="""
    function item_should_remove()
        local pid = photo_id or 0;
        local duration = duration_ms or -1;
        local upload_t = upload_time or 0;
        local now_in_ms = os.time() * 1000;
        local ms_per_day = 24 * 60 * 60 * 1000;
        local long_term_photo = long_term_photo or 0;

        if ((duration < 0) or (pid <= 0)) then
            return true
        end

        if (upload_t < now_in_ms - 180 * ms_per_day) then
            return true
        end

        if (upload_t < now_in_ms - 30 * ms_per_day and long_term_photo == 0) then
            return true
        end

        local count = (explore_stat__real_show_count or 0) + (thanos_stats__real_show_count or 0) + (nebula_stats__real_show_count or 0)
        if count < 100 then
            return true
        else
            return false
        end
    end
    """
)

knn_retr = mio_data.retrieve_from(
    dest_bucket=photo_index,
    enable_auto_calc=True,
    enable_precision_eval=True,
    cache_expire_second=300, 
    cache_max_item_num=500000, 
)

class GpuKnnRetrFlow(LeafFlow):
    def retrieve(self, **kwargs):
        return (
            self
            .limit(0)
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
                import_item_attr=["ann_score"],
                export_item_attr=["ann_score_value"],
                function_for_item="calculate",
                lua_script="""
                function calculate()
                    return ann_score[1];
                end
                """,
            )
            .copy_item_meta_info(save_item_id_to_attr="ann_pid")
            .filter_by_attr(
                attr_name="ann_pid",
                remove_if="<=",
                compare_to=0,
                remove_if_attr_missing=True,
            )
            .filter_by_attr(
                attr_name="ann_score_value",
                remove_if="<=",
                compare_to=0.0,
                remove_if_attr_missing=True,
            )
            .deduplicate()
            .pack_item_attr(
                item_source={
                    "reco_results": True,
                },
                mappings=[
                    {
                        "aggregator": "concat",
                        "to_common_attr": "sim_photo_ids",
                    },
                    {
                        "aggregator": "concat",
                        "from_item_attr": "ann_score_value",
                        "to_common_attr": "sim_photo_scores",
                    },
                ],
            )
            .gen_common_attr_by_lua(
                attr_map={
                    "sim_photo_id_num": "#(sim_photo_ids or {})",
                    "sim_photo_score_num": "#(sim_photo_scores or {})",
                }
            )
            .perflog_attr_value(
                check_point=f"{identifier}.result",
                common_attrs=[
                    "sim_photo_id_num",
                    "sim_photo_score_num",
                    "i2i_ann_topk",
                ],
            )
        )


flow = GpuKnnRetrFlow("gpu_knn_retr").retrieve()
service = LeafService(
    kess_name=kess,
    common_attrs_from_request=["photo_id_list", "photo_emb_list", "i2i_ann_topk"],
    ann_config=knn_retr.get_config(),
)

service.IGNORE_NO_SOURCE_ATTR = ["item_key"]
LeafService.CHECK_UNUSED_ATTR = False
service.return_common_attrs(attrs=["sim_photo_ids", "sim_photo_scores"])
service.add_leaf_flows(request_type="gpu_knn", leaf_flows=[flow])

if __name__ == "__main__":
    service.build(
        output_file=__file__.replace(".py", ".json"),
    )
