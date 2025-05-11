#!/usr/bin/env python3
# coding=utf-8

import os, sys

from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../dragon/tools/pypi/"))
sys.path.append('/home/root/dragon_kbuild/dragon/tools/pypi/')
from dragonfly.common_leaf_dsl import LeafService, IndexSource
from tower_public import *

config_root_dir = os.path.join(os.path.dirname(__file__))


emb_kess = "grpc_point_09b_new_code_base"
emb_shards = 16
queue_prefix = "list_09b_new_code_base"
history_view_num = 256
model_key = "point_09b_new_code_base"
reason = 1024
perf_name = "point_09b_new_code_base"
colossusdb_embd_service_name = "hqg_one_rec"
colossusdb_embd_table_name = "emb_09b_point_simple_base"
EOL = '\n'

abtest_config = {}

class GenerativeRetrFlow(TowerPredictFlowBase,PDNApiMixin, CofeaApiMixin,UniPredictApiMixin):
    def get_ab_param(self):
        return self.get_abtest_params(**abtest_config,)

    def get_kconf(self):
        kconf_configs_var = {
            "enable_nearline_cache": 0,
            "nearline_cache_expire_times": 300,
            "nearline_cache_key_prefix": model_key,
            "label_prompt": 0.0,
            "reset_reason": 4375,
        }

        retrieval_kconf = [
            {"kconf_key": "rinf.rlRunner.Generative_TIGER_config", "export_common_attr": k, "json_path": f"{model_key}.{k}", "default_value": v}
            for k, v in kconf_configs_var.items()
        ]

        self.get_kconf_params(kconf_configs=retrieval_kconf)

        return self

    def duration_perf(self):
        return self.get_item_attr_by_local_attr_index(attrs=["duration_ms"]) \
            .pack_item_attr(
                item_source={"reco_results": True,},
                mappings=[{"from_item_attr": "duration_ms", "to_common_attr": "duration_list"}]
            ) \
            .if_("duration_list ~= nil") \
            .enrich_attr_by_lua(
                import_common_attr=["duration_list"],
                export_common_attr=[
                    "total_num",
                    "item_duration_0",
                    "item_duration_1_3",
                    "item_duration_4_7",
                    "item_duration_8_12",
                    "item_duration_13_20",
                    "item_duration_21_58",
                    "item_duration_58_90",
                    "item_duration_90_120",
                    "item_duration_120_180",
                    "item_duration_180_300",
                    "item_duration_300_420",
                    "item_duration_420_600",
                    "item_duration_gt_600",
                ],
                function_for_common="calculate",
                lua_script="""
                function calculate()
                    local total_num = 0
                    local num_0 = 0
                    local num_1_3 = 0
                    local num_4_7 = 0
                    local num_8_12 = 0
                    local num_13_20 = 0
                    local num_21_58 = 0
                    local num_58_90 = 0
                    local num_90_120 = 0
                    local num_120_180 = 0
                    local num_180_300 = 0
                    local num_300_420 = 0
                    local num_420_600 = 0
                    local num_gt_600 = 0

                    for i = 1, #duration_list do
                        if duration_list[i] ~= nil then
                            total_num = total_num + 1

                            if(duration_list[i] <= 0) then
                                num_0 = num_0 + 1
                            end
                            if(duration_list[i] > 0 and duration_list[i] <= 3000) then
                                num_1_3 = num_1_3 + 1
                            end
                            if(duration_list[i] > 3000 and duration_list[i] <= 7000) then
                                num_4_7 = num_4_7 + 1
                            end
                            if(duration_list[i] > 7000 and duration_list[i] <= 12000) then
                                num_8_12 = num_8_12 + 1
                            end
                            if(duration_list[i] > 12000 and duration_list[i] <= 20000) then
                                num_13_20 = num_13_20 + 1
                            end
                            if(duration_list[i] > 20000 and duration_list[i] <= 58000) then
                                num_21_58 = num_21_58 + 1
                            end
                            if(duration_list[i] > 58000 and duration_list[i] <= 90000) then
                                num_58_90 = num_58_90 + 1
                            end
                            if(duration_list[i] > 90000 and duration_list[i] <= 120000) then
                                num_90_120 = num_90_120 + 1
                            end
                            if(duration_list[i] > 120000 and duration_list[i] <= 180000) then
                                num_120_180 = num_120_180 + 1
                            end
                            if(duration_list[i] > 180000 and duration_list[i] <= 300000) then
                                num_180_300 = num_180_300 + 1
                            end
                            if(duration_list[i] > 300000 and duration_list[i] <= 420000) then
                                num_300_420 = num_300_420 + 1
                            end
                            if(duration_list[i] > 420000 and duration_list[i] <= 600000) then
                                num_420_600 = num_420_600 + 1
                            end
                            if(duration_list[i] > 600000) then
                                num_gt_600 = num_gt_600 + 1
                            end
                        end
                    end
                    return total_num, num_0/total_num, num_1_3/total_num, num_4_7/total_num, num_8_12/total_num, num_13_20/total_num, num_21_58/total_num, num_58_90/total_num, num_90_120/total_num, num_120_180/total_num, num_180_300/total_num, num_300_420/total_num, num_420_600/total_num, num_gt_600/total_num
                end
                """
            ) \
            .perflog_attr_value(
                check_point="generative.duration",
                common_attrs=[
                    "total_num",
                    "item_duration_0",
                    "item_duration_1_3",
                    "item_duration_4_7",
                    "item_duration_8_12",
                    "item_duration_13_20",
                    "item_duration_21_58",
                    "item_duration_58_90",
                    "item_duration_90_120",
                    "item_duration_120_180",
                    "item_duration_180_300",
                    "item_duration_300_420",
                    "item_duration_420_600",
                    "item_duration_gt_600",
                ],
            ) \
            .end_()

    def hitrate_perf(self, namespace, subtag, service_name):
        return self \
            .perflog_attr_value(check_point="generative.hit_rate", common_attrs=["hit_rate_1", "hit_rate_2", "hit_rate_3"]) \
            .gen_common_attr_by_lua(attr_map={"tab_id_str": "tostring(tab_id)"}) \
            .perflog(mode="interval", value="{{hit_rate_1}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_1", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_2}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_2", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_3}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_3", extra3="{{tab_id_str}}") \
            .if_("hit_rate_1 > 0") \
                .perflog(mode="interval", value="{{rank_index_1}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_1", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_2 > 0") \
                .perflog(mode="interval", value="{{rank_index_2}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_2", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_3 > 0") \
                .perflog(mode="interval", value="{{rank_index_3}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_3", extra3="{{tab_id_str}}") \
            .end_()

    def retrieve(self, **kwargs):
        model_config = load_mio_tf_tower_model(
            config_root_dir,
            "tiger",
            "onerec_predict",
            kwargs.get("feature_type", "mio"),
        )

        # emb fetcher 配置
        user_fetcher_slots_config = []
        for c in model_config.slots_config:
            sc = dict()
            sc['input_name'] = c['input_name']
            sc['slots'] = c['slots']
            sc['dtype'] = 'mio_int16'
            #sc['dtype'] = 'scale_int8'
            sc['expand'] = c.get('expand', 1)
            sc['dim'] = c['dim']
            sc['sized'] = 1 if c.get('sized', False) else 0
            if c.get('common', False):
                sc['common'] = True
                user_fetcher_slots_config.append(sc)

        return (
            self.enrich_general_colossus_feature(
                colossus_service_name="grpc_colossusRecoSimItemV3",
                colossus_resp_attr="colossus_resp",
                colossus_output_type="common_item",
                output_slot_attr="colossus_common_slots",
                output_sign_attr="colossus_common_signs",
                output_item_num_attr="colossus_item_num",
                output_pids_attr="colossus_browsed_ids",
                output_channels_attr="colossus_channels",
                colossus_item_limit=10000,
                filter_channels=[],
                filter_ev=False,
                filter_lv=False,
                filter_pos=False,
                filter_future_ts=True,
                extract_item_limit=10000,
            ).enrich_general_colossus_feature(
                colossus_service_name="grpc_colossusRecoSimItemV3",
                colossus_resp_attr="colossus_resp",
                colossus_output_type="common_item",
                output_slot_attr="colossus_common_slots",
                output_sign_attr="colossus_common_signs",
                output_item_num_attr="colossus_item_num",
                output_pids_attr="tmp_colossus_pids",
                output_channels_attr="colossus_channels",
                colossus_item_limit=10000,
                filter_channels=[],
                filter_ev=False,
                filter_lv=True,
                filter_pos=False,
                filter_future_ts=True,
                extract_item_limit=10000,
            )
            .enrich_attr_by_lua(
                import_common_attr=["tmp_colossus_pids"],
                export_common_attr=["history_ev_list","history_ev_len"],
                function_for_common="cal_his_ev",
                lua_script=f"""
                function cal_his_ev()
                    local his_ev_sign = tmp_colossus_pids or {{}}
                    local history_ev_list = {{}}
                    local history_view_num = {history_view_num}
                    history_view_num = math.min(history_view_num, #his_ev_sign)

                    for i=1, history_view_num do
                        --local pid = his_ev_sign[history_view_num-i+1]
                        local pid = his_ev_sign[i]
                        table.insert(history_ev_list, pid)
                    end

                    return history_ev_list, history_view_num
                    end
                """
            )
            .enrich_attr_by_lua(
                import_common_attr = ["history_ev_list","history_ev_len","user_id"],
                export_common_attr = ["common_slots","common_parameters"],
                function_for_common = "extract_sign",
                lua_script="""
                function extract_sign()
                    local common_slots = {}
                    local common_parameters = {}
                    local MIO_MASK_60 = (1<<60)-1

                    for i=1,#(history_ev_list or {}) do
                        local pid_sign = (1<<60) | (MIO_MASK_60 & history_ev_list[i])
                        table.insert(common_slots,1)
                        table.insert(common_parameters,pid_sign)
                    end

                    --local user_sign = (4<<60) | (MIO_MASK_60 & user_id)
                    --table.insert(common_slots,4)
                    --table.insert(common_parameters,user_sign)
                    return common_slots, common_parameters
                    end
                """
            )
            .uni_predict_fused(
                graph=model_config.graph,
                optimizers=["FuseMioVariableCast"],
                key=queue_prefix,
                queue_prefix=queue_prefix,
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
                outputs=[dict(attr_name=attr_name, tensor_name=tensor_name, common=True) for attr_name, tensor_name in model_config.outputs],
                param=model_config.param,
                model_loader_config=dict(rowmajor=True,
                                        type="MioTFExecutedByTensorFlowModelLoader",
                                        implicit_batch=False,
                                        executor_batchsizes=[1],  # user 单边预估，batch_size 为 1
                                        receive_dnn_model_as_macro_block=True,
                                        enable_xla=False,
                                        enable_fp16=True),
                batching_config=dict(batch_timeout_micros=0,
                                    max_batch_size=1,
                                    max_enqueued_batches=1,
                                    batch_task_type="BasicBatchingTask"),
                executor_config=dict(intra_op_parallelism_threads_num=4,
                                    inter_op_parallelism_threads_num=4,
                                    memory_size_per_context=1536*2,
                                    context_per_device=6),
                embedding_fetchers=[dict(fetcher_type="ColossusdbEmbeddingServerFetcher",
                                    colossusdb_embd_model_name=colossusdb_embd_service_name,
                                    colossusdb_embd_table_name=colossusdb_embd_table_name,
                                    shards=emb_shards,
                                    client_side_shard=True,
                                    max_signs_per_request=1000,
                                    thread_num=32,
                                    timeout_ms=5,
                                    common_slots_inputs=["common_slots"],
                                    common_parameters_inputs=["common_parameters"],
                                    slots_config=user_fetcher_slots_config)]
            )
            .enrich_attr_by_lua(
                import_common_attr=["decoder_result_output", "prob_result_output"],
                export_common_attr=["sid1_list", "sid2_list", "sid3_list", "item_num",
                    "prob1_list", "prob2_list", "prob3_list",],
                function_for_common="func",
                lua_script="""
                    function func()
                        local sid1 = {}
                        local sid2 = {}
                        local sid3 = {}

                        local prob1 = {}
                        local prob2 = {}
                        local prob3 = {}

                        for i = 1, #decoder_result_output do
                            if i%3 == 1 then
                                table.insert(sid1, math.floor(decoder_result_output[i]))
                                table.insert(prob1, prob_result_output[i])
                            elseif i%3 == 2 then
                                table.insert(sid2, math.floor(decoder_result_output[i]))
                                table.insert(prob2, prob_result_output[i])
                            elseif i%3 == 0 then
                                table.insert(sid3, math.floor(decoder_result_output[i]))
                                table.insert(prob3, prob_result_output[i])
                            end
                        end
                        return sid1, sid2, sid3, math.floor(#decoder_result_output/3), prob1, prob2, prob3
                    end
                """
            )
            .fake_retrieve(num="{{item_num}}")
            .dispatch_common_attr(from_common_attr="sid1_list", to_item_attr="sid1")
            .dispatch_common_attr(from_common_attr="sid2_list", to_item_attr="sid2")
            .dispatch_common_attr(from_common_attr="sid3_list", to_item_attr="sid3")
            .enrich_attr_by_lua(
                import_item_attr=["sid1", "sid2", "sid3"],
                export_item_attr=["sign"],
                function_for_item="func",
                lua_script="""
                function func()
                    local semantic_code_str = tostring(sid1)..'.'..tostring(sid2)..'.'..tostring(sid3)
                    local sign = util.CityHash64(semantic_code_str)
                    return sign
                end
                """
            )
            .get_remote_embedding_lite_v2(
                kess_service="ia-semantic-ids-code2id-server-v6",
                shard_num=2,
                id_converter=dict(type_name="plainIdConverter"),
                slot=1,
                query_source_type="item_attr",
                input_attr_name="sign",
                client_side_shard=True,
                output_attr_name="pid",
                timeout_ms=20,
                is_raw_data=True,
                raw_data_type='uint64',
                size=1,
            )
            .enrich_attr_by_lua(
                import_item_attr=["pid"],
                export_item_attr=["item_id"],
                function_for_item="func",
                lua_script="""
                    function func()
                        if pid == nil then 
                            return {0}
                        end
                        return pid
                    end
                """
            )
            .pack_item_attr(
                item_source={ "reco_results": True },
                mappings=[{"from_item_attr": "item_id", "to_common_attr": "retrieval_photo_id_list", }]
            )
            .limit(0)
            .retrieve_by_common_attr(attr="retrieval_photo_id_list", reason=reason)
            .copy_attr(
                attrs=[{
                    "from_common": "reset_reason",
                    "to_item": "reset_reason"
                }]
            )
            .set_reason_by_item_attr(reason_attr='reset_reason')
            .dispatch_common_attr(from_common_attr="sid1_list", to_item_attr="sid1")
            .dispatch_common_attr(from_common_attr="sid2_list", to_item_attr="sid2")
            .dispatch_common_attr(from_common_attr="sid3_list", to_item_attr="sid3")
            .dispatch_common_attr(from_common_attr="prob1_list", to_item_attr="prob1")
            .dispatch_common_attr(from_common_attr="prob2_list", to_item_attr="prob2")
            .dispatch_common_attr(from_common_attr="prob3_list", to_item_attr="prob3")
            .dispatch_common_attr(from_common_attr="all_evtr_pred", to_item_attr="all_evtr")
            .dispatch_common_attr(from_common_attr="all_lvtr_pred", to_item_attr="all_lvtr")
            .dispatch_common_attr(from_common_attr="ltr_pred", to_item_attr="ltr")
            .dispatch_common_attr(from_common_attr="wtr_pred", to_item_attr="wtr")
            .dispatch_common_attr(from_common_attr="cmtr_pred", to_item_attr="cmtr")
            .dispatch_common_attr(from_common_attr="setr_pred", to_item_attr="setr")
            .dispatch_common_attr(from_common_attr="ftr_pred", to_item_attr="ftr")
            .dispatch_common_attr(from_common_attr="cltr_pred", to_item_attr="cltr")
            .dispatch_common_attr(from_common_attr="left_slide_pred", to_item_attr="left_slide")
            .dispatch_common_attr(from_common_attr="cmef_pred", to_item_attr="cmef")
            .dispatch_common_attr(from_common_attr="epst_pred", to_item_attr="epst")
            .enrich_attr_by_lua(
                import_item_attr=["sid1", "sid2", "sid3", "prob1", "prob2", "prob3",],
                export_item_attr=["sid_list", "log_p"],
                function_for_item='func',
                lua_script="""
                function func()
                    local sid_list = {}
                    table.insert(sid_list, sid1)
                    table.insert(sid_list, sid2)
                    table.insert(sid_list, sid3)
                    return sid_list, math.log(prob1*prob2*prob3)
                end
                """
            )
            .copy_item_meta_info(save_item_id_to_attr="origin_pid")
            .filter_by_attr(
                attr_name="origin_pid",
                remove_if="<=",
                compare_to=0,
                remove_if_attr_missing=True
            )
        )

    def perf_reward_value(self, namespace, subtag, service_name):
        perf_pxtrs = ["evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", "vtr", "svr", "ptr", "htr", "lsst", "wtd_v2", "cpr"]
        perf_pxtrs += ["wtd_finish_score", "session_play_time", "qtr", "dtr", "epstr", "cmef", "cltr", "adp_wtd", "playlet_ctr", "setr2"]
        perf_pxtrs += ["f1_score"]
        min_pxtrs = ["svr"]
        self.pack_item_attr(
            item_source={ "reco_results": True },
            mappings=[
                {"from_item_attr": x, "to_common_attr": x+"_MAX", "aggregator": "max"} for x in perf_pxtrs
            ] + [
                {"from_item_attr": x, "to_common_attr": x+"_AVG", "aggregator": "avg"} for x in perf_pxtrs
            ] + [
                {"from_item_attr": x, "to_common_attr": x+"_MIN", "aggregator": "min"} for x in min_pxtrs
            ]
        )

        self.sort_by("log_p") \
        .copy_item_meta_info(
            save_item_seq_to_attr="item_seq"
        ) \
        .pack_item_attr(
            item_source={ "reco_results": True },
            mappings=[
                {"from_item_attr": x, "to_common_attr": x+"_TOP6_AVG", "aggregator": "avg"} for x in perf_pxtrs
            ],
            target_item = { "item_seq": [0, 1, 2, 3, 4, 5] }
        ) \
        .enrich_attr_by_lua(
            import_common_attr=[f"{x}_TOP6_AVG" for x in perf_pxtrs],
            export_common_attr=[f"{x}_TOP6_AVG" for x in perf_pxtrs],
            function_for_common="calc",
            lua_script=f"""
            function calc()
                {EOL.join(f"local {x} = math.floor({x}_TOP6_AVG*10000)" for x in perf_pxtrs)}
                return {", ".join(x for x in perf_pxtrs)}
            end
            """
        )

        self.enrich_attr_by_lua(
            import_common_attr=[f"{x}_MAX" for x in perf_pxtrs],
            export_common_attr=[f"{x}_MAX" for x in perf_pxtrs],
            function_for_common="calc",
            lua_script=f"""
            function calc()
                {EOL.join(f"local {x} = math.floor({x}_MAX*10000)" for x in perf_pxtrs)}
                return {", ".join(x for x in perf_pxtrs)}
            end
            """
        )

        self.enrich_attr_by_lua(
            import_common_attr=[f"{x}_AVG" for x in perf_pxtrs],
            export_common_attr=[f"{x}_AVG" for x in perf_pxtrs],
            function_for_common="calc",
            lua_script=f"""
            function calc()
                {EOL.join(f"local {x} = math.floor({x}_AVG*10000)" for x in perf_pxtrs)}
                return {", ".join(x for x in perf_pxtrs)}
            end
            """
        )

        self.enrich_attr_by_lua(
            import_common_attr=[f"{x}_MIN" for x in min_pxtrs],
            export_common_attr=[f"{x}_MIN" for x in min_pxtrs],
            function_for_common="calc",
            lua_script=f"""
            function calc()
                {EOL.join(f"local {x} = math.floor({x}_MIN*10000)" for x in min_pxtrs)}
                return {", ".join(x for x in min_pxtrs)}
            end
            """
        )

        for x in perf_pxtrs:
            self.perflog(
                mode="interval",
                value="{{" + x+"_MAX" + "}}",
                namespace=namespace,
                subtag=subtag,
                extra1=service_name,
                extra2=x+"_MAX",
                extra3="{{tab_id_str}}"
            )

            self.perflog(
                mode="interval",
                value="{{" + x+"_AVG" + "}}",
                namespace=namespace,
                subtag=subtag,
                extra1=service_name,
                extra2=x+"_AVG",
                extra3="{{tab_id_str}}"
            )

            self.perflog(
                mode="interval",
                value="{{" + x+"_TOP6_AVG" + "}}",
                namespace=namespace,
                subtag=subtag,
                extra1=service_name,
                extra2=x+"_TOP6_AVG",
                extra3="{{tab_id_str}}"
            )
        
        for x in min_pxtrs:
            self.perflog(
                mode="interval",
                value="{{" + x+"_MIN" + "}}",
                namespace=namespace,
                subtag=subtag,
                extra1=service_name,
                extra2=x+"_MIN",
                extra3="{{tab_id_str}}"
            )
        return self

# 第一步，读取 kconf 配置
get_kconf_ab_flow = (GenerativeRetrFlow(name="get_kconf_ab_flow")
    .get_kconf()
    .copy_user_meta_info(
        save_user_id_to_attr="user_id",
        save_device_id_to_attr="device_id",
    )
)

# 第二步，近线召回，从 redis 读取结果
read_nearline_flow = (GenerativeRetrFlow(name="read_nearline_flow")
    .if_("disable_nearline_cache == nil")
        .set_attr_value(common_attrs=[{"name": "disable_nearline_cache", "type": "string", "value": "0"}])
    .end_()
    .gen_common_attr_by_lua(attr_map={"disable_nearline_cache": "tonumber(disable_nearline_cache)"}) 
    .if_("user_id ~= 0 and enable_nearline_cache == 1 and disable_nearline_cache == 0")
        .str_format(
            format_string="%s_%lu",
            input_attrs=["nearline_cache_key_prefix", "user_id"],
            output_attr="nearline_cache_redis_key",
            is_common_attr=True
        )
        .retrieve_by_redis(
            reason=reason,
            retrieve_num=1000,
            cluster_name="tdmRecoTest",
            key_from_attr="nearline_cache_redis_key",
            item_separator=","
        )
        .count_reco_result(save_count_to="current_item_num")
        .if_("current_item_num ~= 0")
            .return_()
        .end_()
    .end_()
)

# 第三步，extract user info
extract_user_flow = (GenerativeRetrFlow(name="extract_user_flow")
    .parse_protobuf_from_string(
        is_common_attr=True,
        ttl_seconds=7200,
        input_attr="user",
        output_attr="user_info",
        class_name="ks::reco::UserInfo",
    )
    .enrich_with_protobuf(
        from_extra_var="user_info",
        is_common_attr=True,
        attrs=[
            # 拼接 browsed_pids
            dict(path="browsed_photo_ids", name="browsed_photo_ids"),
            dict(
                path="user_profile_v1.real_show_list.photo_id",
                name="real_show_list",
            ),
            dict(
                path="user_profile_v1.click_list.photo_id",
                name="click_list",
            ),
            dict(
                path="user_profile_v1.like_list.photo_id",
                name="like_list",
            ),
            dict(
                path="nearby_browsed_photo_ids",
                name="nearby_browsed_photo_ids",
            ),
            dict(
                path="slide_browsed_photo_ids",
                name="slide_browsed_photo_ids",
            ),
        ],
    )
    .pack_common_attr(
        input_common_attrs=[
            "browsed_photo_ids",
            "real_show_list",
            "click_list",
            "like_list",
            "nearby_browsed_photo_ids",
            "slide_browsed_photo_ids",
        ],
        output_common_attr="browsed_pids",
    )
)

# 第四步，请求 colossus
colossus_flow = (GenerativeRetrFlow(name="colossus_flow")
    .colossus(
        service_name="grpc_colossusRecoSimItemV3",
        client_type="common_item_client",
        output_attr="colossus_resp",
        parse_to_pb=False,
        max_resp_item_num=10000,
    )
)

# 第五步，前向 infer
retr_flow = GenerativeRetrFlow(name="retr_flow").retrieve()

# 第六步，browse set 过滤
post_filter_flow = (GenerativeRetrFlow(name="post_filter_flow")
    .if_("browsed_pids ~= nil")
    .filter_by_common_attr(common_attr=["browsed_pids"])
    .end_()
    .filter_by_common_attr(common_attr=["colossus_browsed_ids"])
)

# 第七步，召回结果写入 nearline redis
write_nearline_flow = (GenerativeRetrFlow(name="write_nearline_flow")
    .if_("enable_nearline_cache == 1 and disable_nearline_cache == 0 and #retrieval_photo_id_list >= 500")
        .write_to_redis(
            kcc_cluster="tdmRecoTest",
            expire_second="{{nearline_cache_expire_times}}",
            key="{{nearline_cache_redis_key}}",
            value="{{retrieval_photo_id_list}}"
        )
    .end_()
)

# 第八步，召回结果分析
perf_flow = (GenerativeRetrFlow(name="perf_flow")
    .gen_common_attr_by_lua(attr_map={"is_perf": "math.random() < 0.05"})
    .if_("is_perf > 0")
        .duration_perf()
    .end_()
)

# eval request 第一步，评估样本处理
pre_eval_flow = (GenerativeRetrFlow(name="pre_eval_flow")
    .if_("eval_pos_photo_id_list ~= nil")
    .retrieve_by_common_attr(attr="eval_pos_photo_id_list", reason=1)
    .copy_item_meta_info(save_item_id_to_attr="eval_pos_pid")
    .get_remote_embedding_lite_v2(
        protocol=1,
        colossusdb_embd_service_name="rlj-24q2-norm-exp",
        colossusdb_embd_table_name="semantic_id_new",
        client_side_shard=True,
        shard_num=2,
        id_converter={"type_name": "plainIdConverter"},
        input_attr_name="eval_pos_pid",
        output_attr_name="eval_semantic_id_list",
        is_raw_data=True,
        query_source_type="item_attr",
        raw_data_type="uint16",
        timeout_ms=100,
        size=3
    )
    .enrich_attr_by_lua(
        import_item_attr=["eval_semantic_id_list"],
        export_item_attr=["non_empty_semantic_id", "eval_layer_1", "eval_layer_2", "eval_layer_3"],
        function_for_item="func",
        lua_script="""
        function func()
            local non_empty_semantic_id = 1
            if eval_semantic_id_list == nil then
                non_empty_semantic_id = 0
            end

            local sid1 = eval_semantic_id_list[1]
            local sid2 = eval_semantic_id_list[2]
            local sid3 = eval_semantic_id_list[3]

            local eval_layer_1;
            local eval_layer_2;
            local eval_layer_3;

            if eval_semantic_id_list ~= nil then
                eval_layer_1 = util.CityHash64(tostring(sid1))
                eval_layer_2 = util.CityHash64(tostring(sid1)..tostring(sid2))
                eval_layer_3 = util.CityHash64(tostring(sid1)..tostring(sid2)..tostring(sid3))
            end
            return non_empty_semantic_id, eval_layer_1, eval_layer_2, eval_layer_3
        end
        """
    ).filter_by_attr(
        attr_name="non_empty_semantic_id",
        remove_if="==",
        compare_to=0,
        remove_if_attr_missing=False
    ).copy_user_meta_info(
        save_result_size_to_attr="eval_length"
    ).perflog_attr_value(
        check_point="generative.eval_length",
        common_attrs=["eval_length"]
    ).pack_item_attr(
        item_source={ "reco_results": True },
        mappings=[
            {"from_item_attr": "eval_layer_1", "to_common_attr": "eval_layer_1_list", },
            {"from_item_attr": "eval_layer_2", "to_common_attr": "eval_layer_2_list", },
            {"from_item_attr": "eval_layer_3", "to_common_attr": "eval_layer_3_list", },
        ]
    )
    .limit(0)
    .end_()
)

post_eval_flow = (GenerativeRetrFlow(name="post_eval_flow")
    .if_("eval_length ~= nil and eval_length > 0")
    .enrich_attr_by_lua(
        import_common_attr=["eval_layer_1_list", "eval_layer_2_list", "eval_layer_3_list",
            "middle_layer_1", "middle_layer_2", "middle_layer_3", "eval_length"],
        export_common_attr=["hit_rate_1", "hit_rate_2", "hit_rate_3", "rank_index_1", "rank_index_2", "rank_index_3",],
        function_for_common="func",
        lua_script="""
        function func()
            -- pre
            local global_dict = {}

            for i=1, #middle_layer_1, 1 do
                sid1 = math.floor(middle_layer_1[i])
                key = util.CityHash64(tostring(sid1))
                global_dict[key] = i
            end

            for i=1, #middle_layer_2, 2 do
                sid1 = math.floor(middle_layer_2[i])
                sid2 = math.floor(middle_layer_2[i+1])
                key = util.CityHash64(tostring(sid1)..tostring(sid2))
                global_dict[key] = (i+1) / 2
            end

            for i=1, #middle_layer_3, 3 do
                sid1 = math.floor(middle_layer_3[i])
                sid2 = math.floor(middle_layer_3[i+1])
                sid3 = math.floor(middle_layer_3[i+2])
                key = util.CityHash64(tostring(sid1)..tostring(sid2)..tostring(sid3))
                global_dict[key] = (i+2) / 3
            end

            local hit_1 = 0;
            local hit_2 = 0;
            local hit_3 = 0;

            local rank_index_1 = 0;
            local rank_index_2 = 0;
            local rank_index_3 = 0;

            -- 第一层 评估
            for i = 1, #eval_layer_1_list do
                if global_dict[eval_layer_1_list[i]] ~= nil then
                    hit_1 = hit_1 + 1
                    rank_index_1 = rank_index_1 + global_dict[eval_layer_1_list[i]]
                end
            end

            -- 第二层 评估
            for i = 1, #eval_layer_2_list do
                if global_dict[eval_layer_2_list[i]] ~= nil then
                    hit_2 = hit_2 + 1
                    rank_index_2 = rank_index_2 + global_dict[eval_layer_2_list[i]]
                end
            end

            -- 第三层 评估
            for i = 1, #eval_layer_3_list do
                if global_dict[eval_layer_3_list[i]] ~= nil then
                    hit_3 = hit_3 + 1
                    rank_index_3 = rank_index_3 + global_dict[eval_layer_3_list[i]]
                end
            end

            rank_index_1 = rank_index_1 / hit_1
            rank_index_2 = rank_index_2 / hit_2
            rank_index_3 = rank_index_3 / hit_3

            return math.floor(hit_1 / eval_length * 10000), math.floor(hit_2 / eval_length * 10000), math.floor(hit_3 / eval_length * 10000), math.floor(rank_index_1*10000), math.floor(rank_index_2*10000), math.floor(rank_index_3*10000)
        end
        """
    )
    .hitrate_perf(namespace="common.leaf", subtag="tiger_hit_rate", service_name=model_key)
    .end_()
)

# 请求最新的精排模型，评估 reward
main_model_pxtrs = [
    "evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", "vtr", "svr", "ptr", "tag_click",
    "join_topic", "vtr_v2", "dtr", "epstr", "lwt", "slide_next_rate", "dfvr", "cmef",
    "etcm", "live", "htr", "qtr", "osftr", "cltr", "setr", "scalar0", "scalar1",
    "scalar2", "linear0", "linear1", "linear2", "switch_out_featured_tab", "video_slide",
    "ptltr", "swptr", "swpst", "swppc", "swpac", "lstr", "lsst", "llvtr", "long_watch_time",
    "mus1", "mus2", "mus3", "vrec", "mfsc", "mus4", "mtm1", "mtm2", "mtm3", "csf", "cmagic",
    "tag_ctr", "tag_vtr", "mustc", "twhc", "vmagic", "browse_depth", "combine_unity",
    "evtr_v3", "lvtr_v3", "vtr_v3", "fr_log_vtr", "bottom_search_pctr", "adaptive_wtd",
    "cpr", "evtr_exp", "lvtr_exp", "svr_exp", "vtr_exp", "kyplc", "post_at_comment_score",
    "screen_shot", "search_comment_highlight_click", "search_comment_highlight_trending",
    "watchlive_wtd_combine", "sppc_bottom_bar", "interest_evtr_playtime", "wtd_duration_score",
    "vtr_calibration_score", "fuse_score", "new_svr", "sppc_combine", "wtd_finish_score",
    "cpr_wtd", "wtd_duration_score_v2", "wtd_v2", "caption_searchpage", "click_live",
    "effective_watch_live_time", "setr3_low_active", "search_pure_cmt_highlight_ctr", "itr",
    "playlet_ctr", "adp_wtd", "hashtag_ctr", "peak_evtr", "search_comment_trending_click",
    "bubble_cr", "hashtag_sppc", "bottom_bar_ctr", "ua_long_term_page_score", "session_play_time",
    "all_evtr", "avtt", "playtime_denoise", "pro_cpr", "total_watch_time", "total_watch_time_wtd",
    "adp_clevtr_pro", "pro_evtr", "watch_live", "wtd_v2_playtime", "cpr_duration_score", "setr2",
    "bubble_sr", "search_page_photo_show", "search_page_photo_click"
]

reward_eval_flow = (GenerativeRetrFlow(name="reward_eval_flow")
    .count_reco_result(save_count_to="item_num")
    .if_("item_num > 0")
    .copy_item_meta_info(
        save_item_key_to_attr="item_id",
    )
    .get_item_attr_by_distributed_index(
        photo_store_kconf_key="reco.distributedIndex.hotPhotoStoreConfig",
        use_dynamic_photo_store=True,
        attrs=[{ "name": "is_living", "path": "live_photo_info.is_living" },]
    )
    .build_protobuf(  # 电商 直播链路在自己的链路里提前 build , 能拿到这些pxtr吗？
        inputs = [
            { "item_attr": "is_living", "path": "living" },
            { "item_attr": "item_id", "path": "ar_result.pid" },
        ],
        output_item_attr = "reco_photo_info_str",
        class_name = "ks::reco::RecoPhotoInfo",
        as_string = True
    )
    .set_attr_value(
        no_overwrite=True,
        common_attrs=[{
          "name": "retr_type",
          "type": "int",
          "value": 1
        },
        {
          "name": "tab_id",
          "type": "int",
          "value": 10000
        },]
    )
    .delegate_enrich(
        name="delegate_enrich_main_model",
        kess_service="grpc_hqg24q4ModelComboFinal",
        request_type="predict_for_gamora",
        timeout_ms=10000,
        send_common_attrs = [
            {"name": "user", "as": "user_info_str"},
            "tab_id",
            "retr_type"
        ],
        send_item_attrs=["reco_photo_info_str"],
        recv_item_attrs=main_model_pxtrs,
        partition_size=256,
        use_packed_item_attr=True
    )
    .calc_by_formula1(
        import_item_attr=main_model_pxtrs,
        kconf_key="formula.scenarioKey58.ll_dpo_train_f1",
        export_formula_value = [
            "f1_score",
        ],
        abtest_biz_name="KUAISHOU_APPS"
    )
    .enrich_attr_by_lua(
        import_item_attr=["f1_score"],
        export_item_attr=["f1_score"],
        function_for_item="calc",
        lua_script="""
            function calc()
                return math.min(f1_score, 10000.0)
            end
        """
    )
    .perflog_attr_value(check_point="default.reward", item_attrs=["evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", "vtr", "svr", "ptr"],)
    .perf_reward_value(namespace="common.leaf", subtag="reward_value_pxtr", service_name=perf_name)
    .end_()
)

kess_name = "kess"

print(f"kess name: {kess_name}")

service = LeafService(
    kess_name=kess_name,
    common_attrs_from_request=["user", "disable_nearline_cache", "eval_pos_photo_id_list", "tab_id"],
    index_source=IndexSource.LOCAL_ATTR_INDEX,
)
service.AUTO_INJECT_ITEM_ATTR = False
service.CHECK_UNUSED_ATTR = False
service.IGNORE_NO_SOURCE_ATTR=["browsed_pids", "all_evtr_pred", "all_lvtr_pred", "ltr_pred", "wtr_pred", "cmtr_pred", "setr_pred", "ftr_pred", "cltr_pred", "left_slide_pred", "cmef_pred", "epst_pred"]
returned_item_attrs = ["sid_list", "prob1", "prob2", "prob3","log_p"] + [
    "all_evtr", "all_lvtr", "ltr", "wtr", "cmtr", "setr", "ftr", "cltr", "left_slide", "cmef", "epst"
]
service.return_item_attrs(attrs=returned_item_attrs)

# 线上请求 flow：get_kconf_ab_flow->read_nearline_flow->extract_user_flow->colossus_flow->retr_flow->post_filter_flow->write_nearline_flow->perf_flow
service.add_leaf_flows(request_type="default", leaf_flows=[colossus_flow, get_kconf_ab_flow, read_nearline_flow, extract_user_flow, retr_flow, post_filter_flow, write_nearline_flow])
service.add_leaf_flows(request_type="eval_request", leaf_flows=[colossus_flow, get_kconf_ab_flow, pre_eval_flow, retr_flow, post_filter_flow, post_eval_flow, reward_eval_flow])

if __name__ == "__main__":
    service.build(
        output_file=os.path.join(
            "./",
            os.path.basename(__file__).replace(".py", "") + ".json",
        )
    )
