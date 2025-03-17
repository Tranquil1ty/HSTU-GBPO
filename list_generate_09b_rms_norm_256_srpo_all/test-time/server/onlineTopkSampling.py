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

# 新的命名方式
emb_kess = "grpc_list09b_sparse"
service_name = "grpc_list09b_predict_test_time"
kess_name = service_name
emb_shards = 16
queue_prefix = "List_generate_09B_backup"
history_view_num = 256
max_target_num = 5
model_key = "List_generate_09B_backup"
code_len = 3
EOL="\n"
abtest_config = {}


# infer_item_num = 9
# beam_search_num = 128

class GenerativeRetrFlow(TowerPredictFlowBase,PDNApiMixin, CofeaApiMixin,UniPredictApiMixin):
    def get_ab_param(self):
        return self.get_abtest_params(**abtest_config,)

    def get_kconf(self):
        kconf_configs_var = {
            "enable_nearline_cache": 0,
            "nearline_cache_expire_times": 300,
            "nearline_cache_key_prefix": service_name,
            "label_prompt": 0.0,
            "sampling_temperature": [1.0, 1.0, 1.0],
            "top_k_sampling_k": [128.0, 128.0, 128.0],
            "return_m_input": 256.0,
            "is_reward_model_on": 1,
            "reward_model_kess_name": "grpc_rljListScoreForGen",
            "reward_model_request_type": "predict_for_gamora",
            "reward_model_timeout": 1000,
            "reward_model_partition_size": 100,
        }

        retrieval_kconf = [
            {"kconf_key": "rinf.rlRunner.List_generate_config", "export_common_attr": k, "json_path": f"{service_name}.{k}", "default_value": v}
            for k, v in kconf_configs_var.items()
        ]

        self.get_kconf_params(kconf_configs=retrieval_kconf)

        # override by param from request
        self.enrich_attr_by_lua(
            import_common_attr=[
                "sampling_temperature", "top_k_sampling_k", "return_m_input",
                "sampling_temperature_from_req", "top_k_sampling_k_from_req", "return_m_from_req",],
            export_common_attr=["sampling_temperature", "top_k_sampling_k", "return_m_input"],
            function_for_common="calculate",
            lua_script="""
                function calculate()
                    local sampling_temperature = sampling_temperature_from_req or sampling_temperature
                    local top_k_sampling_k = top_k_sampling_k_from_req or top_k_sampling_k
                    local return_m = {math.floor(return_m_from_req or return_m_input) * 1.0}
                    local return_top_k_sampling_k = {}
                    for i=1,#top_k_sampling_k do
                        table.insert(return_top_k_sampling_k, math.floor(top_k_sampling_k[i])*1.0)
                    end
                    return sampling_temperature, return_top_k_sampling_k, return_m
                end
            """)

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
            .perflog_attr_value(check_point="generative.hit_rate", common_attrs=["hit_rate_1", "hit_rate_2", "hit_rate_3", "hit_rate_4", "hit_rate_5", "hit_rate_6"]) \
            .gen_common_attr_by_lua(attr_map={"tab_id_str": "tostring(tab_id)"}) \
            .perflog(mode="interval", value="{{hit_rate_1}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_1", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_2}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_2", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_3}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_3", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_4}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_4", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_5}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_5", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_6}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="hit_rate_6", extra3="{{tab_id_str}}") \
            .if_("hit_rate_1 > 0") \
                .perflog(mode="interval", value="{{rank_index_1}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_1", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_2 > 0") \
                .perflog(mode="interval", value="{{rank_index_2}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_2", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_3 > 0") \
                .perflog(mode="interval", value="{{rank_index_3}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_3", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_4 > 0") \
                .perflog(mode="interval", value="{{rank_index_4}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_4", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_5 > 0") \
                .perflog(mode="interval", value="{{rank_index_5}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_5", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_6 > 0") \
                .perflog(mode="interval", value="{{rank_index_6}}", namespace=namespace, subtag=subtag, extra1=service_name, extra2="rank_index_6", extra3="{{tab_id_str}}") \
            .end_()

    def perf_reward_value(self, namespace, subtag, service_name):
        self.enrich_attr_by_lua(
            import_item_attr=return_pxtrs,
            export_item_attr=[f"{x}_double" for x in return_pxtrs],
            function_for_item="calc",
            lua_script=f"""
            function calc()
                {EOL.join(f"local {x} = {x}*1.0" for x in return_pxtrs)}
                return {", ".join(x for x in return_pxtrs)}
            end
            """
        )

        self.enrich_attr_by_lua(
            import_item_attr=["f1_score"],
            export_item_attr=["f1_score", "f1_score_double"],
            function_for_item="calc",
            lua_script=f"""
            function calc()
                return math.floor(f1_score * 10000), f1_score*10000.0
            end
            """
        ) \
        .pack_item_attr(
            item_source={ "reco_results": True },
            mappings=[
                {"from_item_attr": "f1_score", "to_common_attr": "f1_score_MAX", "aggregator": "max"},
                {"from_item_attr": "f1_score_double", "to_common_attr": "f1_score_AVG", "aggregator": "avg"}
            ]
        )

        self.pack_item_attr(
            item_source={ "reco_results": True },
            mappings=[
                {"from_item_attr": x, "to_common_attr": x+"_MAX", "aggregator": "max"} for x in return_pxtrs
            ] + [
                {"from_item_attr": x+"_double", "to_common_attr": x+"_AVG", "aggregator": "avg"} for x in return_pxtrs
            ]
        )

        for x in return_pxtrs:
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
            value="{{f1_score_MAX}}",
            namespace=namespace,
            subtag=subtag,
            extra1=service_name,
            extra2="f1_score_MAX",
            extra3="{{tab_id_str}}"
        )
        self.perflog(
            mode="interval",
            value="{{f1_score_AVG}}",
            namespace=namespace,
            subtag=subtag,
            extra1=service_name,
            extra2="f1_score_AVG",
            extra3="{{tab_id_str}}"
        )

        return self

    def retrieve(self, **kwargs):
        model_config = load_mio_tf_tower_model(
            config_root_dir,
            "tiger",
            "predictTopkSampling",
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

                    local user_sign = (4<<60) | (MIO_MASK_60 & user_id)
                    table.insert(common_slots,4)
                    table.insert(common_parameters,user_sign)
                    return common_slots, common_parameters
                    end
                """
            )
            .enrich_attr_by_lua(
                import_common_attr=["label_prompt", "extra_label_prompt_input"],
                export_common_attr=["label_prompt_input"],
                function_for_common="func",
                lua_script="""
                function func()
                    local label_prompt_input = {}
                    if extra_label_prompt_input == nil then
                        table.insert(label_prompt_input, label_prompt)
                    else
                        table.insert(label_prompt_input, extra_label_prompt_input)
                    end
                    return label_prompt_input
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
                ] + [{
                    "attr_name": "label_prompt_input",
                    "tensor_name": "label_prompt_input",
                    "dim": 1,
                    "common": True,
                },
                {
                    "attr_name": "sampling_temperature",
                    "tensor_name": "sampling_temperature",
                    "dim": code_len,
                    "common": True,
                },
                {
                    "attr_name": "top_k_sampling_k",
                    "tensor_name": "top_k_sampling_k",
                    "dim": code_len,
                    "common": True,
                },
                {
                    "attr_name": "return_m_input",
                    "tensor_name": "return_m_input",
                    "dim": 1,
                    "common": True,
                }],
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
                embedding_fetchers=[dict(fetcher_type="BtEmbeddingServerFetcher",
                                    kess_service=emb_kess,
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
                export_common_attr=["sid1_list", "sid2_list", "sid3_list", "item_num", "prob1_list", "prob2_list", "prob3_list"],
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
            .enrich_attr_by_lua(
                import_common_attr=["item_num"],
                export_common_attr=["reason_list"],
                function_for_common="func_reason",
                lua_script=f"""
                    function func_reason()
                        local max_target_num = {max_target_num}
                        local len = item_num // max_target_num
                        local reason_list = {{}}
                        
                        for i=1, len do
                            for j=1, max_target_num do
                                table.insert(reason_list, i)
                            end
                        end
                        return reason_list
                    end
                """
            )
            .dispatch_common_attr(from_common_attr="sid1_list", to_item_attr="sid1")
            .dispatch_common_attr(from_common_attr="sid2_list", to_item_attr="sid2")
            .dispatch_common_attr(from_common_attr="sid3_list", to_item_attr="sid3")
            .dispatch_common_attr(from_common_attr="prob1_list", to_item_attr="prob1")
            .dispatch_common_attr(from_common_attr="prob2_list", to_item_attr="prob2")
            .dispatch_common_attr(from_common_attr="prob3_list", to_item_attr="prob3")
            .enrich_attr_by_lua(
                import_item_attr=["sid1", "sid2", "sid3", "prob1", "prob2", "prob3"],
                export_item_attr=["sid_list", "log_p", "prob_list"],
                function_for_item='func',
                lua_script="""
                function func()
                    local sid_list = {}
                    local prob_list = {}
                    table.insert(sid_list, sid1)
                    table.insert(sid_list, sid2)
                    table.insert(sid_list, sid3)
                    table.insert(prob_list, prob1)
                    table.insert(prob_list, prob2)
                    table.insert(prob_list, prob3)
                    return sid_list, math.log(prob1*prob2*prob3), prob_list
                end
                """
            )
            .enrich_attr_by_lua(
                import_item_attr=["sid1", "sid2", "sid3"],
                export_item_attr=["sign"],
                function_for_item="func",
                lua_script="""
                function func()
                    local semantic_code_str = tostring(sid1)..tostring(sid2)..tostring(sid3)
                    local sign = util.CityHash64(semantic_code_str)
                    return sign
                end
                """
            )
            .get_remote_embedding_lite_v2(
                kess_service="ia-semantic-ids-code2id-server-v4",
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
                mappings=[
                    {"from_item_attr": "item_id", "to_common_attr": "retrieval_photo_id_list", },
                    {"from_item_attr": "log_p", "to_common_attr": "log_p_list", },
                    {"from_item_attr": "sid_list", "to_common_attr": "sid_list_list", },
                ]
            )
            .limit(0)
            .copy_user_meta_info(save_request_type_to_attr="request_type")
            .if_("request_type == 'end2end' or request_type == 'eval_request'")
                .gen_common_attr_by_lua(
                    attr_map={ "fake_item_num": f"math.floor(#retrieval_photo_id_list / {max_target_num})", }
                )
                .fake_retrieve(num="{{fake_item_num}}")
                .dispatch_common_attr(
                    dispatch_config = [
                        {
                        "from_common_attr" : "retrieval_photo_id_list",
                        "to_item_attr" : "session_photo_id_list",
                        "by_list_size": max_target_num,
                        },{
                        "from_common_attr" : "log_p_list",
                        "to_item_attr" : "session_log_p",
                        "by_list_size": max_target_num,
                        }
                    ]
                )
                .enrich_attr_by_lua(
                    import_item_attr=["session_log_p"],
                    export_item_attr=["session_score"],
                    function_for_item="func",
                    lua_script="""
                    function func()
                        local session_score = 0
                        for i=1, #session_log_p do
                            session_score = session_score + session_log_p[i]
                        end
                        return session_score
                    end
                    """
                )
                .sort(score_from_attr="session_score")
            .else_()
                .retrieve_by_common_attr(attr="retrieval_photo_id_list", reason=1024)
                .dispatch_common_attr(from_common_attr="sid_list_list", to_item_attr="sid_list", by_list_size=3)
                .copy_item_meta_info(save_item_id_to_attr="origin_pid_default")
                .filter_by_attr(
                    attr_name="origin_pid_default",
                    remove_if="<=",
                    compare_to=0,
                    remove_if_attr_missing=True
                )
            .end_()
        )

# 第一步，读取 kconf 配置
get_kconf_ab_flow = (GenerativeRetrFlow(name="get_kconf_ab_flow")
    .get_kconf()
    .copy_user_meta_info(
        save_user_id_to_attr="user_id",
        #save_device_id_to_attr="device_id",
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
            reason=1025,
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

# eval request
session_eval_flow = (GenerativeRetrFlow(name="session_eval_flow")
    .if_("eval_pos_photo_id_list ~= nil")
        .copy_user_meta_info(save_result_size_to_attr="candidate_session_len")
        .if_("candidate_session_len == 0")
            .return_()
        .end_()
        .enrich_attr_by_lua(
            import_common_attr=["eval_pos_photo_id_list"],
            import_item_attr=["session_photo_id_list"],
            export_item_attr=["session_eval_rate"],
            function_for_item="compute_eval",
            lua_script="""
            function compute_eval()
                local eval_len = #eval_pos_photo_id_list;
                local session_set = {};
                local hit_cnt = 0;

                -- 去重
                for i = 1, #session_photo_id_list do
                    if session_set[session_photo_id_list[i]] == nil then
                        session_set[session_photo_id_list[i]] = 1
                    end
                end

                for k, v in pairs(session_set) do
                    for j = 1, #eval_pos_photo_id_list do
                        if eval_pos_photo_id_list[j] == k then
                            hit_cnt = hit_cnt + 1
                            break
                        end
                    end
                end

                return math.floor(hit_cnt / eval_len * 10000)
            end
            """
        )
        .pack_item_attr(
            item_source={ "reco_results": True },
            mappings=[
                {"from_item_attr": "session_eval_rate", "to_common_attr": "max_session_eval_rate", "aggregator": "max"},
                {"from_item_attr": "session_eval_rate", "to_common_attr": "sum_session_eval_rate", "aggregator": "sum"},
                {"from_item_attr": "session_eval_rate", "to_common_attr": "session_eval_rate_list" },
                {"from_item_attr": "session_photo_id_list", "to_common_attr": "final_retr_list"},
            ]
        )
        .enrich_attr_by_lua(
            import_common_attr=["session_eval_rate_list", "max_session_eval_rate"],
            export_common_attr=["max_session_eval_rate_index"],
            function_for_common="func",
            lua_script="""
                function func()
                    local max_index = -1;
                    for i=1, #session_eval_rate_list do
                        if session_eval_rate_list[i] == max_session_eval_rate then
                            max_index = i;
                            break;
                        end
                    end
                    return max_index
                end
            """
        )
        .gen_common_attr_by_lua(attr_map={"avg_session_eval_rate": "math.floor(sum_session_eval_rate / candidate_session_len)"})
        .enrich_attr_by_lua(
            import_common_attr=["eval_pos_photo_id_list", "final_retr_list"],
            export_common_attr=["final_eval_rate", "final_retr_len"],
            function_for_common="compute_eval",
            lua_script="""
            function compute_eval()
                local eval_len = #eval_pos_photo_id_list;
                local retr_set = {};
                local hit_cnt = 0;
                local final_retr_len = 0;

                -- 去重
                for i = 1, #final_retr_list do
                    if retr_set[final_retr_list[i]] == nil then
                        retr_set[final_retr_list[i]] = 1
                    end
                end

                for k, v in pairs(retr_set) do
                    final_retr_len = final_retr_len + 1
                    for j = 1, #eval_pos_photo_id_list do
                        if eval_pos_photo_id_list[j] == k then
                            hit_cnt = hit_cnt + 1
                            break
                        end
                    end
                end

                return math.floor(hit_cnt / eval_len * 10000), final_retr_len
            end
            """
        )
        .perflog_attr_value(check_point="generative.hit_rate", common_attrs=["max_session_eval_rate", "avg_session_eval_rate", "final_eval_rate", "final_retr_len"]) \
        .gen_common_attr_by_lua(attr_map={"tab_id_str": "tostring(tab_id)"}) \
        .perflog(mode="interval", value="{{max_session_eval_rate}}", namespace="common.leaf", subtag="session_grm_model", extra1=service_name, extra2="max_session_eval_rate", extra3="{{tab_id_str}}") \
        .perflog(mode="interval", value="{{avg_session_eval_rate}}", namespace="common.leaf", subtag="session_grm_model", extra1=service_name, extra2="avg_session_eval_rate", extra3="{{tab_id_str}}") \
        .perflog(mode="interval", value="{{final_eval_rate}}", namespace="common.leaf", subtag="session_grm_model", extra1=service_name, extra2="final_eval_rate", extra3="{{tab_id_str}}") \
        .perflog(mode="interval", value="{{final_retr_len}}", namespace="common.leaf", subtag="session_grm_model", extra1=service_name, extra2="final_retr_len", extra3="{{tab_id_str}}") \
        .perflog(mode="interval", value="{{max_session_eval_rate_index}}", namespace="common.leaf", subtag="session_grm_model", extra1=service_name, extra2="max_index", extra3="{{tab_id_str}}") \
    .end_()
)

# 奖励模型评估
return_pxtrs = ["seq_wtd", "seq_ltr", "seq_cmtr", "seq_wtr", "seq_ftr", "seq_evtr", "seq_vtr", "seq_swt"]
reward_model_eval_flow = (GenerativeRetrFlow(name="reward_model_eval_flow")
    .if_("request_type == 'eval_request'")
        .if_("is_reward_model_on == 1")
            .enrich_attr_by_lua(
                import_item_attr=["session_photo_id_list"],
                export_item_attr=["zero_filter_valid_item"],
                function_for_item="calc",
                lua_script=f"""
                    function calc()
                        local pid = -1
                        if #session_photo_id_list ~= {max_target_num} then
                        return 0
                        end 
                        for i = 1, #session_photo_id_list do
                        pid = session_photo_id_list[i]
                        if pid == 0 then
                            return 0
                        end
                        end
                        return 1
                    end
                """
            )
            .filter_by_rule(
                rule = {
                    "attr_name": "zero_filter_valid_item",
                    "remove_if": "==",
                    "compare_to": 0
                }
            )
            .count_reco_result(save_count_to="list_after_zero_filter_num") \
            .pack_item_attr_to_item_attr(
                from_item_attrs=["session_photo_id_list"],
                to_item_attr="dedup_session_photo_id_list",
                dedup_to_item_attr=True,
                default_val=[0,0,0,0,0]
            )
            .enrich_attr_by_lua(
                import_item_attr=["dedup_session_photo_id_list"],
                export_item_attr=["is_repeated_session_item"],
                function_for_item="calc",
                lua_script=f"""
                    function calc()
                        if #dedup_session_photo_id_list ~= {max_target_num} then
                            return 1
                        end
                        return 0
                    end
                """
            )
            .filter_by_rule(
                rule = {
                    "attr_name": "is_repeated_session_item",
                    "remove_if": "==",
                    "compare_to": 1
                }
            )
            .count_reco_result(target_item={"is_repeated_session_item" : 0}, save_count_to="list_after_repeat_filter_num")
            .enrich_attr_by_lua(
                target_item={"is_repeated_session_item" : 0},
                import_item_attr=["session_photo_id_list"],
                export_item_attr=["is_not_in_browset_session"],
                import_common_attr=["colossus_browsed_ids"],
                function_for_item="calc",
                lua_script="""
                    function calc()
                        local browset_map = {}
                        local is_not_in_browset_session = 1

                        for i = 1, #colossus_browsed_ids do
                            if browset_map[colossus_browsed_ids[i]] == nil then
                                browset_map[colossus_browsed_ids[i]] = 1
                            end
                        end

                        for j = 1, #session_photo_id_list do
                            if browset_map[session_photo_id_list[j]] == 1 then
                                is_not_in_browset_session = 0
                                break
                            end
                        end

                        return is_not_in_browset_session
                    end
                """
            )
            .count_reco_result(target_item={"is_not_in_browset_session" : 1}, save_count_to="list_after_browset_filter_num")
            .filter_by_rule(
                rule = {
                    "attr_name": "is_not_in_browset_session",
                    "remove_if": "==",
                    "compare_to": 0
                }
            )
            .set_attr_value(
                no_overwrite=True,
                common_attrs=[
                    { "name": "retr_type", "type": "int", "value": 1 },
                    { "name": "is_eval", "type": "int", "value": 1 },
                ]
            )
            .delegate_enrich(
                kess_service="{{reward_model_kess_name}}",
                request_type="{{reward_model_request_type}}",
                timeout_ms="{{reward_model_timeout}}",
                send_item_attrs=["session_photo_id_list"],
                send_common_attrs = [
                    "retr_type",
                    "is_eval"
                ],
                recv_item_attrs=[f"{x}_list" for x in return_pxtrs],
                partition_size="{{reward_model_partition_size}}",
                use_packed_item_attr = True
            )
            .enrich_attr_by_lua(
                import_item_attr=[f"{x}_list" for x in return_pxtrs],
                export_item_attr=return_pxtrs,
                function_for_item="calc",
                lua_script=f"""
                function calc()
                    {EOL.join(f"local {x} = 0" for x in return_pxtrs)}
                    {EOL.join(f"if #{x}_list > 0 then {x} = {x}_list[1] end" for x in return_pxtrs)}
                    return {", ".join(x for x in return_pxtrs)}
                end
                """
            )
            .calc_by_formula1(
                kconf_key="formula.scenarioKey81.MUW_gen_reco",
                export_formula_value = [
                    "f1_score"
                ],
                abtest_biz_name="KUAISHOU_APPS"
            )
            .enrich_attr_by_lua(
                import_item_attr=return_pxtrs,
                export_item_attr=return_pxtrs,
                function_for_item="calc",
                lua_script=f"""
                function calc()
                    {EOL.join(f"local {x} = math.floor({x}*10000)" for x in return_pxtrs)}
                    return {", ".join(x for x in return_pxtrs)}
                end
                """
            )
            .perf_reward_value(namespace="common.leaf", subtag="reward_value", service_name=service_name)
            .perflog_attr_value(check_point="reward.seqPxtr", common_attrs=["list_after_zero_filter_num", "list_after_repeat_filter_num", "list_after_browset_filter_num"], item_attrs=[f"{x}" for x in return_pxtrs])
        .end_()
    .end_()
)

print(f"kess name: {kess_name}")

service = LeafService(
    kess_name=kess_name,
    common_attrs_from_request=["user", "disable_nearline_cache", "eval_pos_photo_id_list", "tab_id", "extra_label_prompt_input",
        "sampling_temperature_from_req", "top_k_sampling_k_from_req", "return_m_from_req"],
    index_source=IndexSource.LOCAL_ATTR_INDEX,
)
service.AUTO_INJECT_ITEM_ATTR = False
#service.CHECK_UNUSED_ATTR = False
service.IGNORE_UNUSED_ATTR = [
    'colossus_channels', 'colossus_common_signs', 'colossus_common_slots', 'colossus_item_num',
    'beam_search_logits_output', 'reason_list', 'colossus_browsed_ids', 'browsed_pids', 'nearline_cache_expire_times',
    "return_m_input_out", "top_k_sampling_k_input_out", "session_photo_id_list", "check_return_m"
]
service.IGNORE_NO_SOURCE_ATTR=["browsed_pids", "session_photo_id_list"]
returned_item_attrs = ["sid_list", "prob_list", "log_p", "session_id", "session_photo_id_list", "session_score"]
service.return_item_attrs(attrs=returned_item_attrs)

# 线上请求 flow：get_kconf_ab_flow->read_nearline_flow->extract_user_flow->colossus_flow->retr_flow->post_filter_flow->write_nearline_flow->perf_flow
service.add_leaf_flows(request_type="default", leaf_flows=[colossus_flow, get_kconf_ab_flow, read_nearline_flow, extract_user_flow, retr_flow, post_filter_flow, write_nearline_flow])
service.add_leaf_flows(request_type="end2end", leaf_flows=[colossus_flow, get_kconf_ab_flow, read_nearline_flow, extract_user_flow, retr_flow])
service.add_leaf_flows(request_type="eval_request", leaf_flows=[colossus_flow, get_kconf_ab_flow, retr_flow, session_eval_flow, reward_model_eval_flow])

if __name__ == "__main__":
    service.build(
        output_file=os.path.join(
            "./",
            os.path.basename(__file__).replace(".py", "") + ".json",
        )
    )
