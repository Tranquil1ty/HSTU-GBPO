#!/usr/bin/env python3
# coding=utf-8

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(filename)s:%(lineno)s %(message)s",
)

from dragonfly.common_leaf_dsl import OfflineRunner, LeafFlow, LeafService
from dragonfly.ext.swing.swing_api_mixin import SwingApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin

ann_kess = "grpc_AnnIAEmb"
kgnn_kess = "grpc_kgnn_i2i_ia-I2I"
identifier = "i2i_ia_ann_config"
kconf_key = f"rinf.rlRunner.{identifier}"
kgnn_shard = 4
timeout = 1500
max_subflow_num = 100

class I2IRunnerFlow(
    LeafFlow, OfflineApiMixin, KgnnApiMixin, SwingApiMixin, KuibaApiMixin, MioApiMixin, GsuApiMixin, EmbedCalcApiMixin, PDNApiMixin, CofeaApiMixin
):
    # NEW_ROUND
    def _retrieve_by_bt_shm_kv(self, **kwargs): 
        bt_shm_kv_timeout_ms = kwargs.get('bt_shm_kv_timeout_ms', 150)
        bt_shm_kv_parts_num = kwargs.get('bt_shm_kv_parts_num', 512000)
        index_service=kwargs.get('index_service', "grpc_hotPhotoInfoServiceOffline")
        index_service_shards = kwargs.get('index_service_shards', 8)
        bt_shm_kv_service=kwargs.get('bt_shm_kv_service', "grpc_hotBTShmKV8sOffline")
        param = dict(
            kess_service=bt_shm_kv_service,
            shard_name=index_service,
            timeout_ms=bt_shm_kv_timeout_ms,
            num_shards=index_service_shards,
            num_parts=bt_shm_kv_parts_num,
            reason=0)
        if "index_use_raw_key" in kwargs: param["use_raw_key"]=kwargs["index_use_raw_key"]

        self.retrieve_from_bt_shm_kv(**param)
        return self

    def _retrieve_by_realtime_photo_index(self, **kwargs):
        param_retrieval = dict(
            batch_size=kwargs.get("realtime_update_batch_size", 1),
            queue_name=kwargs.get("realtime_index_btqueue_name", "reco_index_builder_hot_photo_realtime"),
            sleep_for_wait_btq_ms=kwargs.get("sleep_for_wait_btq_ms", 100),
            save_item_info_to_attr=kwargs.get("save_item_info_to_attr", "photo_info"),
            enable_lite_photo_map=kwargs.get("enable_lite_photo_map", True),
            photo_map_capacity=kwargs.get("lite_photo_map_capacity", 10000000),
            photo_upload_expire_sec=kwargs.get("lite_photo_upload_expire_sec", 1800),
            just_run_at_shard0=False,
            reason=1,
        )
        self.retrieve_photo_info_from_btq(**param_retrieval)
        return self

    def _retrieve_by_lite_photo_map(self, **kwargs):
        batch_size = kwargs.get("lite_update_batch_size", 1) #计算 photo embedding 的 batch size 
        update_interval_sec= kwargs.get("lite_update_interval_sec", 1)
        photo_map_capacity = kwargs.get("lite_photo_map_capacity", 10000000)
        photo_upload_expire_sec = kwargs.get("lite_photo_upload_expire_sec", 1800)

        self.retrieve_photo_from_lite_photo_map(
            batch_size=batch_size,
            update_interval_sec=update_interval_sec,
            photo_map_capacity=photo_map_capacity,
            photo_upload_expire_sec=photo_upload_expire_sec,
            reason=2,
        )
        return self
    
    def _retrieve_by_missed_trigger(self, **kwargs):
        self.fetch_message(
            group_id='test',
            btq_prefix='missed_trigger_pid',
            btq_batch_size = 1,
            output_attr ='trigger_string'
        ) \
        .enrich_attr_by_lua(
            import_common_attr = ["trigger_string"],
            export_common_attr = ["photo_id_list"],
            function_for_common = "stringToIntArray",
            lua_script = """
            function stringToIntArray()
                local photo_id_list = {}
                for pid_str in string.gmatch(trigger_string, "[^,]+") do
                    local pid = tonumber(pid_str)
                    if pid then
                        table.insert(photo_id_list, pid)
                    end
                end
                return photo_id_list
            end
            """
        )
        return self

    def _prepare(self):
        self.get_kconf_params(
                kconf_configs = [{
                    "kconf_key": kconf_key,
                    "export_common_attr": "i2i_ann_topk",
                    "json_path": "ann_retrieve_num",
                    "default_value": 50
                }]
            )
        return self

    def _get_item_index_attr(self): #获取有效视频信息
        self \
        .get_item_attr_by_distributed_index( #获取PhotoInfo，只支持主站精选
            photo_store_kconf_key="reco.distributedIndex.hotPhotoStoreConfig",
            use_dynamic_photo_store=True,
            attrs=[ #将 PhotoInfo 中的字段作为 ItemAttr 进行保存
                "photo_id",
                "upload_time",
                "duration_ms",
                {"path": "explore_stat.real_show_count", "name": "e_vv_cnt"},
                {"path": "nebula_stats.real_show_count", "name": "n_vv_cnt"},
                {"path": "thanos_stats.real_show_count", "name": "g_vv_cnt"},
                "long_play_count",
                "show_count",
            ]
        ) \
        .enrich_attr_by_lua( #过滤无效视频
            import_item_attr=[
                "e_vv_cnt",
                "n_vv_cnt",
                "g_vv_cnt",
                "photo_id",
                "duration_ms",
                "upload_time",
            ],
            export_item_attr=[
                "is_remove",
            ],
            function_for_item="calc",
            lua_script="""
            function calc()
                local is_remove = 0;
                local e_vv = e_vv_cnt or 0;
                local n_vv = n_vv_cnt or 0;
                local g_vv = g_vv_cnt or 0;
                local total_vv = n_vv + g_vv + e_vv;
                local duration = duration_ms or -1;
                local pid = photo_id or -1;
                local upload_t = upload_time or 0;
                local now_in_ms = os.time() * 1000;
                local ms_per_day = 24 * 60 * 60 * 1000;
                
                if ((pid < 0) or (duration < 0) or (total_vv < 100)) then
                    is_remove = 1;
                end 
                -- if (upload_t < now_in_ms - 30 * ms_per_day) then
                -- is_remove = 1; 
                -- end 
                return is_remove;
            end
            """
        )
        self.log_debug_info(
            for_debug_request_only=True,
            item_attrs=[
                "photo_id",
                "duration_ms",
                "g_vv_cnt",
                "n_vv_cnt",
                "e_vv_cnt",
                "photo_id",
                "is_remove",
            ]
        )
        self.filter_by_attr(
            attr_name="is_remove",
            remove_if="==",
            compare_to=1,
            remove_if_attr_missing=True,
        )
        self.count_reco_result(
            save_count_to = "item_num"
        ) 
        self.if_("item_num <= 0").return_(0, "no item_num").end_()
        self.pack_item_attr(
            item_source = {
                "reco_results": True,
            },
            mappings = [{
                "from_item_attr": "photo_id",
                "to_common_attr": "photo_id_list",
            }]
        )
        return self

    def _select_lv_item(self):
        # 选择长播视频
        select_lv_config = {
            "select_lv_rate_threshold": 0.1,
            "select_show_count_threshold": 1000,
        }
        self.get_kconf_params( 
            kconf_configs=[
                {
                    "kconf_key": kconf_key,
                    "export_common_attr": k,
                    "json_path": k,
                    "default_value": v,
                } for k, v in select_lv_config.items()
            ]
        )
        # 过滤低质量视频
        self.enrich_attr_by_lua( 
            import_item_attr=["long_play_count", "show_count"],
            import_common_attr = ["select_lv_rate_threshold", "select_show_count_threshold"],
            export_item_attr=["is_lv", "lv_rate"],
            function_for_item="calc",
            lua_script="""
            function calc()
                local is_lv = 1;
                local lv_rate = long_play_count/show_count;
                if (lv_rate < select_lv_rate_threshold) or (show_count < select_show_count_threshold) then
                    is_lv = 0;
                end 
                return is_lv, lv_rate;
            end
            """
        )
        # 统计长播率
        self.pack_item_attr(
            item_source = {
                "reco_results": True,
            },
            mappings = [{
                "aggregator": "sum",
                "from_item_attr": "is_lv",
                "to_common_attr": "is_lv_count",
            }]
        )
        self.gen_common_attr_by_lua(
            attr_map={
                    "is_lv_rate": "is_lv_count/item_num",
                }
        )
        self.log_debug_info(
            for_debug_request_only=True,
            item_attrs=["long_play_count", "show_count", "is_lv", "lv_rate"],
            common_attrs = ["is_lv_count", "is_lv_rate"]
        )
        self.perflog_attr_value(
            check_point=f"{self.name}.select_lv_item",
            item_attrs=["lv_rate"],
            common_attrs=["is_lv_rate"]
        )
        return self

    def _send_btq_embedding(self, btq_config):
        # send btq of embedding
        self.if_("item_num <= 0").return_(0, "no item_num").end_()
        self._select_lv_item()
        self.generate_update_message(
            target_item = {
                "is_lv": 1
            },
            inputs=[
                dict(embedding_attr="photo_emb", sign_attr="photo_id", slot=btq_config["slot"])
            ],
            shards=btq_config["shard_num"],
            id_converter={"type_name": "mioEmbeddingIdConverter"},
            use_raw_embedding=False,
            expire_seconds=3600 * 24 * 2,
            save_result_to_common_attr="update_message",
            compress_type = "mio_int16"
        )
        self.send_with_btq(
            common_attr=f"update_message0",
            queue_name=f"{btq_config['queue_prefix']}0",
        ) 
        self.log_debug_info(
            for_debug_request_only=True,
            common_attrs=["item_num",] + [f"update_message{shard}" for shard in range(btq_config["shard_num"])],
        )
        return self

    def _get_remote_embedding(self, emb_server_config=None):
        # request embedding server 
        self.limit(0)
        self.retrieve_by_common_attr(
            attr="photo_id_list",
            reason=1,
        )
        self.deduplicate()
        self.copy_item_meta_info(
            save_item_id_to_attr="photo_id",
        )
        self.get_remote_embedding_lite_v2(
            kess_service = emb_server_config["kess_service"],
            shard_num = emb_server_config["shard_num"],
            id_converter=dict(type_name="plainIdConverter"),
            query_source_type = "item_attr",
            input_attr_name = "photo_id",
            output_attr_name = "photo_emb",
            client_side_shard = True,
            timeout_ms = 200,
            size = emb_server_config["emb_size"],
            is_raw_data=True,
            raw_data_type="float32",
        )
        self.filter_by_attr(
            attr_name = "photo_emb",
            remove_if_attr_missing = True
        ) 
        self.count_reco_result(
            save_count_to = "item_num"
        ) 
        self.if_("item_num <= 0").return_(0, "no item_num").end_()
        self.copy_item_meta_info(
            save_item_seq_to_attr="item_seq",
        )
        return self
    
    def _get_sub_flow_id(self):
        # 获取 subflow 的数量，每个 subflow 预计分配 20 个 item, subflow 数量不超过 max_subflow_num
        self.get_kconf_params(
                kconf_configs = [{
                    "kconf_key": kconf_key, 
                    "export_common_attr": 'max_subflow_num',
                    "json_path": 'max_subflow_num',
                    "default_value": max_subflow_num,
                }]
            )\
            .gen_common_attr_by_lua(
                attr_map={"sub_flow_num": "math.min(max_subflow_num, math.floor(item_num / 20)+1)"}
                )
        # 获取每个 item 的 sub_flow_id
        self.enrich_attr_by_lua(
            function_for_item = "get_sub_flow_id",
            import_item_attr = ["item_seq"],
            import_common_attr = ["sub_flow_num"],
            export_item_attr = ["sub_flow_id"],
            lua_script = """
                function get_sub_flow_id()
                    return item_seq % sub_flow_num
                end
            """
            )
        return self
    
    def _executor_sub_retr_flow(self, subflow_dict):
        # 执行 subflow
        for i in range(max_subflow_num):
            self.retrieve_by_sub_flow(
                sub_flow = subflow_dict[i], 
                pass_all_items=True,
                pass_item_attrs = ['sub_flow_id', 'photo_emb']) 
        return self

class Retr_and_Write_Subflow(
    LeafFlow, OfflineApiMixin, KgnnApiMixin, SwingApiMixin, KuibaApiMixin, MioApiMixin, GsuApiMixin, EmbedCalcApiMixin, PDNApiMixin, CofeaApiMixin
):
    def _retrieve_ann_and_write(self, target_sub_flow_id):
        self.count_reco_result(target_item = {"sub_flow_id": target_sub_flow_id},save_count_to="sub_item_count") 
        self.if_("sub_item_count <= 0").return_(0, "no item_num").end_()
        self.pack_item_attr(
            target_item = {"sub_flow_id": target_sub_flow_id},
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "aggregator": "concat",
                    "to_common_attr": "sub_photo_id_list",
                },
                {
                    "aggregator": "concat",
                    "from_item_attr": "photo_emb",
                    "to_common_attr": "sub_photo_emb_list",
                },
            ],
            )
        self.limit(0)\
        .copy_item_meta_info(target_item =  {"sub_flow_id": target_sub_flow_id},save_item_id_to_attr="item_id") \
        .copy_attr(
            target_item =  {"sub_flow_id": target_sub_flow_id},attrs=[{"from_item": "item_id", "to_common": "photo_id"}]
        ) 
        self.if_(f"sub_photo_id_list ~= nil and #(sub_photo_id_list or {{}}) > 0")
        self.delegate_retrieve( 
            kess_service=ann_kess,
            send_common_attrs=[{"name": f"sub_photo_id_list", "as": "photo_id_list"}, {"name": f"sub_photo_emb_list", "as": "photo_emb_list"}, "i2i_ann_topk"], 
            recv_item_attrs=["ann_pid", "filtered_ann_score", "filtered_src_item"],
            request_type="default",
            request_num = 1000,
            timeout_ms=timeout,
        ) \
       .update_inner_item( #向 kgnn 的某个 relation 更新图存储的边信息
                src_attr="filtered_src_item",
                dst_attr="ann_pid",
                kess_service=kgnn_kess,
                relation_name="I2I",
                dst_weight_attr="filtered_ann_score",
                insert_edge_weight_act=2,
                timeout_ms=1000,
                shard_num=kgnn_shard
            ) \
            .update_inner_item( #向 kgnn 的某个 relation 更新图存储的边信息
                src_attr="ann_pid",
                dst_attr="filtered_src_item",
                kess_service=kgnn_kess,
                relation_name="I2I",
                dst_weight_attr="filtered_ann_score",
                insert_edge_weight_act=2,
                timeout_ms=1000,
                shard_num=kgnn_shard
            ) 
        self.end_()
        return self

emb_server_config = {
  "kess_service": "grpc_mmu_e2e_i2i_PsCloud_online",
  "shard_num": 32,
  "emb_size": 512, 
}


subflow_dict = [Retr_and_Write_Subflow('sub_flow_'+str(i))._retrieve_ann_and_write(i) for i in range(max_subflow_num)]

# 实时发现页索引 BTQ
realtime_update_flow = I2IRunnerFlow(name="realtime_update_flow") \
    ._prepare()\
    ._retrieve_by_realtime_photo_index(realtime_update_batch_size = 20) \
    ._get_item_index_attr() \
    ._get_remote_embedding(emb_server_config=emb_server_config)\
    ._get_sub_flow_id()\
    ._executor_sub_retr_flow(subflow_dict)

# 分布式索引服务 bt_shm_kv
bt_shm_update_flow = I2IRunnerFlow(name="bt_shm_update_flow") \
    ._prepare()\
    ._retrieve_by_bt_shm_kv() \
    ._get_item_index_attr() \
    ._get_remote_embedding(emb_server_config=emb_server_config)\
    ._get_sub_flow_id()\
    ._executor_sub_retr_flow(subflow_dict)

# 最近三十分钟上传的视频
lite_photo_map_update_flow = I2IRunnerFlow(name="lite_photo_map_update_flow") \
    ._prepare()\
    ._retrieve_by_lite_photo_map(lite_update_batch_size = 20) \
    ._get_item_index_attr() \
    ._get_remote_embedding(emb_server_config=emb_server_config)\
    ._get_sub_flow_id()\
    ._executor_sub_retr_flow(subflow_dict)

# retr_server 未命中的视频
missed_trigger_flow = I2IRunnerFlow(name="missed_trigger_flow") \
    ._prepare()\
    ._retrieve_by_missed_trigger() \
    ._get_remote_embedding(emb_server_config=emb_server_config)\
    ._get_sub_flow_id()\
    ._executor_sub_retr_flow(subflow_dict)

LeafService.CHECK_UNUSED_ATTR = False
runner = OfflineRunner("i2i_ia_emb_runner")

runner.add_leaf_flows(leaf_flows=[realtime_update_flow], name="realtime_update_flow", thread_num=128)
runner.add_leaf_flows(leaf_flows=[bt_shm_update_flow], name="bt_shm_update_flow", thread_num=128)
runner.add_leaf_flows(leaf_flows=[lite_photo_map_update_flow], name="lite_photo_map_update_flow", thread_num=8)
runner.add_leaf_flows(leaf_flows=[missed_trigger_flow], name="missed_trigger_flow", thread_num=16)


runner.build(__file__.replace(".py", ".json"))
