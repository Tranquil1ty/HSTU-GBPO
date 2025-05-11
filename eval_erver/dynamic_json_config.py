import os
import sys
import argparse
import numpy as np

from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.arrow.arrow_api_mixin import ArrowApiMixin

sys.path.append("..")
#from direct_retrieval_flow import return_pxtrs

current_dir = os.path.dirname(__file__)
parser = argparse.ArgumentParser()
parser.add_argument("--run", dest="run", default=False, action="store_true")
parser.add_argument("--eval", dest="eval", default=False, action="store_true")
args = parser.parse_args()
return_pxtrs = ["seq_reward", "seq_wtd", "seq_ltr", "seq_cmtr", "seq_wtr", "seq_ftr", "wtd_v2", "evtr", "seq_evtr", "seq_vtr", "seq_swt", "seq_cpr", "seq_dtr", "seq_htr", "seq_svr", "seq_lvtr", "seq_epstr", "seq_ptr", "seq_cmef", "seq_lsst" ]
max_target_num = 5
EOL = "\n"

class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, GsuApiMixin, EmbedCalcApiMixin, KuibaApiMixin, PDNApiMixin, CofeaApiMixin, KgnnApiMixin, ArrowApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)

labels = [
  "realshow",
  "profile",
  "click",
  "like",
  "follow",
  "forward",
  "feedback_negative",
  "comment",
  "playing_time",
  "slide_enter",
  "down_load",
  "left_slide",
  "collect",

  # extended
  "long_view",
  "effective_view",
  "wtd_effective_view"
]

# 第一步，消费无 Lag 采样流量
read_log = (
    DataReaderFlow(name="read_log")
    .fetch_message(
        output_attr="record_batch_in",
        group_id="reco_gen_model_eval",
        kafka_topic="kaiworks_reco_log_sample_batchrow"
    )
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.model2.reco_gen_model_eval_stream_config",
        "value_type": "json",
        "json_path": "sample_rate",
        "export_common_attr": "sample_rate",
      }]
    )
    .enrich_attr_by_lua(
      function_for_common = "calculate",
      import_common_attr = ["sample_rate"],
      export_common_attr = ["rand", "skip"],
      lua_script = """
        function calculate(seq, item_key, reason, score)
          rand = util.Random()
          skip = 0
          if rand > sample_rate then
            skip = 1
          end
          return rand, skip
        end
      """
    )
    .if_("skip == 1")
      .return_(0)
    .end_if_()
    .retrieve_from_batch_row(
        batch_row_from_attr="record_batch_in",
        id_feature="photo_id",
        user_id_feature="user_id",
        device_id_feature="device_id",
        request_time_feature="time_ms",
        extract_common_features=["reco_log_bytes"],
        extract_item_features=[
            "photo_id",
            "reco_reco_hot_label_llsid_pid_bytes",
            "reco_reco_hot_label_uid_pid_bytes",
            "reco_reco_hot_label_did_pid_bytes",
            "reco_reco_hot_label_uid_aid_bytes"
        ],
        save_result_to_common_attr="photo_ids_before_join"
    )
    .perflog(
           mode="interval",
           value="{{_REQ_TIME_}}",
           namespace="common.leaf",
           subtag="gen_eval",
           extra1="eval_time_ms"
    )
    .parse_protobuf_from_string(
        input_attr="reco_log_bytes",
        output_attr="ks_reco_log",
        class_name="ks::reco::RecoLog"
    )
    .joint_reco_log(
        reco_log_attr="ks_reco_log",
        label_llsid_pid_str="reco_reco_hot_label_llsid_pid_bytes",
        label_uid_pid_str="reco_reco_hot_label_uid_pid_bytes",
        label_did_pid_str="reco_reco_hot_label_did_pid_bytes",
        label_uid_aid_str="reco_reco_hot_label_uid_aid_bytes",
        item_list_from_attr="photo_ids_before_join"
    )
    .enrich_with_protobuf(
        from_extra_var="ks_reco_log",
        is_common_attr=True,
        attrs=[
        dict(path="user", name="user_info"),
        dict(path="tab", name="tab_id"),
        dict(path="time", name="time_ms"),
        "llsid", "session_id", "block", "thanos_flag", "traffic_type", "source_photo_info"]
    )
    .retrieve_from_ks_reco_log(
        from_extra_var="ks_reco_log",
        save_reco_photo_to="reco_photo_info"
    )
   .enrich_with_protobuf(
        from_extra_var="reco_photo_info",
        is_common_attr=False,
        attrs=[
            dict(path="photo", name="photo_info"),
            dict(path="partial_stid.st_reco_id", name="st_reco_id_str"),
            "context_info",
        ]
    )
    .enrich_with_protobuf(
        from_extra_var="photo_info",
        is_common_attr=False,
        attrs=[
            "photo_id", "duration_ms", "upload_type","user_hash_tag_id", "upload_time", "picture_type", "ad_native_tag_info", "reco_playlet_tag", "collection_score"
        ]
    ) 
    .enrich_with_protobuf(
        from_extra_var="context_info",
        is_common_attr=False,
        attrs=[
            dict(path="real_show", name="realshow"),
        ]
    )
    .enrich_with_json(
        import_attr="st_reco_id_str",
        attrs = [
            dict(name="st_reco_id", path="mix", output_type="string"),
        ],
        is_common_attr=False
    )
    .enrich_attr_by_lua(
        import_item_attr=["st_reco_id"],
        export_item_attr=["is_consume_stid"],
        function_for_item="compare",
        lua_script="""
            function compare()
                local stid_sub = string.sub(st_reco_id, 1, 2)
                return stid_sub == "1_"
            end
        """
    )
    .serialize_protobuf_message(
        from_common_attr = "user_info",
        serialize_to_common_attr = "user_info_str"
    )
    .if_("tab_id ~= 10000")
        .return_()
    .end_()
    .copy_user_meta_info(save_result_size_to_attr="item_num")
    .if_("item_num == 0")
        .return_()
    .end_()
    .get_kconf_params(kconf_configs=[{
        "kconf_key": "reco.model.GRMNoLogEvalList",
        "export_common_attr": "common_eval_kess_list",
        "value_type": "list_string",
        "default_value": []
    }])
    .enrich_attr_by_lua(
        import_common_attr=["common_eval_kess_list"],
        export_common_attr=["common_eval_kess_list_size",
                            "common_eval_kess_list_index",
                            "common_eval_kess_name",
                            "is_keep_call_eval"],
        function_for_common="calculate",
        lua_script = """
        function calculate()
            local common_eval_kess_list_size = #common_eval_kess_list
            local common_eval_kess_list_index = 1
            local is_keep_call_eval = common_eval_kess_list_index <= common_eval_kess_list_size
            local common_eval_kess_name = ""
            if (is_keep_call_eval) then
                common_eval_kess_name = common_eval_kess_list[common_eval_kess_list_index]
            end
            return common_eval_kess_list_size,
                    common_eval_kess_list_index,
                    common_eval_kess_name,
                    is_keep_call_eval
        end
        """
    )
    .pack_item_attr(
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "servershow_photo_id_list", }],
    )
    .pack_item_attr(
        target_item={ "realshow": 1 },
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "realshow_photo_id_list", }],
    )
    .pack_item_attr(
        target_item={ "is_consume_stid": 1},
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "servershow_consume_photo_id_list", }],
    )
    .pack_item_attr(
        target_item={ "realshow": 1, "is_consume_stid": 1},
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "realshow_consume_photo_id_list", }],
    )
    .truncate(size_limit=0)
)

# 第二步, 构造评估样本 & 请求下游进行评估
class CallEvalServer(LeafFlow):
    def __init__(self, name, loop_if, loop_limit):
        LeafFlow.__init__(self, name=name, loop_if=loop_if, loop_limit=loop_limit)

    def eval_realshow(self, eval_list_name):
        return (self
            .retrieve_by_common_attr(attr=eval_list_name, reason=1024)
            .copy_user_meta_info(save_result_size_to_attr="reco_item_num")
            .if_(f"reco_item_num >= {max_target_num}")
            .copy_item_meta_info(save_item_id_to_attr="photo_id")
            .truncate(size_limit=max_target_num) 
            .pack_item_attr(
                item_source={ "reco_results": True, },
                mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "trunc_photo_id_list", }],
            ) 
            .truncate(size_limit=1) 
            .copy_attr(
                attrs=[{
                    "from_common": "trunc_photo_id_list",
                    "to_item": "session_photo_id_list"
                }]
            ) 
            .set_attr_value(
              common_attrs=[
                {
                  "name": "common_eval_kess_name",
                  "type": "string",
                  "value": eval_list_name
                }
              ]
            ) 
            .count_reco_result(save_count_to="eval_list_num") 
            .delegate_enrich(
                kess_service = "grpc_rljListScoreForGen",
                request_type = "eval_request",
                timeout_ms = 1000,
                send_item_attrs=[
                    "session_photo_id_list"
                ],
                send_common_attrs = [
                    "user_info_str", "tab_id", 
                ],
                recv_item_attrs=[x+"_list" for x in return_pxtrs],
                partition_size = 100,
                use_packed_item_attr = True
            )
            .log_reward()
            .truncate(size_limit=0)
            .end_if_()
        )

    def self_perf(self, attr_names, namespace="common.leaf", subtag="reward_value_eval"):
        self.gen_common_attr_by_lua(attr_map={"tab_id_str": "tostring(tab_id)"})
        for attr_name in attr_names:
            self.perflog(
                   mode="interval",
                   value="{{" + attr_name + "}}",
                   namespace=namespace,
                   subtag=subtag,
                   extra1="{{common_eval_kess_name}}",
                   extra2=attr_name,
                   extra3="{{tab_id_str}}"
            )
        return self

    def filter(self):
        return self.enrich_attr_by_lua(
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
        ) \
        .filter_by_rule(
            rule = {
                "attr_name": "zero_filter_valid_item",
                "remove_if": "==",
                "compare_to": 0
            }
        ) \
        .count_reco_result(save_count_to="eval_list_num")

    def log_recall(self):
      return (self.
        if_("realshow_photo_id_list ~= nil")
          .copy_user_meta_info(save_result_size_to_attr="candidate_session_len")
          .if_("candidate_session_len == 0")
              .return_()
          .end_()
          .enrich_attr_by_lua(
              import_common_attr=["realshow_photo_id_list"],
              import_item_attr=["session_photo_id_list"],
              export_item_attr=["session_eval_rate"],
              function_for_item="compute_eval",
              lua_script="""
              function compute_eval()
                  local eval_len = #realshow_photo_id_list;
                  local session_set = {};
                  local hit_cnt = 0;

                  -- 去重
                  for i = 1, #session_photo_id_list do
                      if session_set[session_photo_id_list[i]] == nil then
                          session_set[session_photo_id_list[i]] = 1
                      end
                  end

                  for k, v in pairs(session_set) do
                      for j = 1, #realshow_photo_id_list do
                          if realshow_photo_id_list[j] == k then
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
              import_common_attr=["realshow_photo_id_list", "final_retr_list"],
              export_common_attr=["final_eval_rate", "final_retr_len"],
              function_for_common="compute_eval",
              lua_script="""
              function compute_eval()
                  local eval_len = #realshow_photo_id_list;
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
                      for j = 1, #realshow_photo_id_list do
                          if realshow_photo_id_list[j] == k then
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
          .perflog(mode="interval", value="{{max_session_eval_rate}}", namespace="common.leaf", subtag="session_grm_model_eval", extra1="{{common_eval_kess_name}}", extra2="max_session_eval_rate", extra3="{{tab_id_str}}") \
          .perflog(mode="interval", value="{{avg_session_eval_rate}}", namespace="common.leaf", subtag="session_grm_model_eval", extra1="{{common_eval_kess_name}}", extra2="avg_session_eval_rate", extra3="{{tab_id_str}}") \
          .perflog(mode="interval", value="{{final_eval_rate}}", namespace="common.leaf", subtag="session_grm_model_eval", extra1="{{common_eval_kess_name}}", extra2="final_eval_rate", extra3="{{tab_id_str}}") \
          .perflog(mode="interval", value="{{final_retr_len}}", namespace="common.leaf", subtag="session_grm_model_eval", extra1="{{common_eval_kess_name}}", extra2="final_retr_len", extra3="{{tab_id_str}}") \
          .perflog(mode="interval", value="{{max_session_eval_rate_index}}", namespace="common.leaf", subtag="session_grm_model_eval", extra1="{{common_eval_kess_name}}", extra2="max_index", extra3="{{tab_id_str}}") \
        .end_()
      )

    def log_reward(self, namespace="common.leaf", subtag="reward_value_eval"):
        return (self
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
          .pack_item_attr(
            item_source = {
              "reco_results": True,
            },
            mappings = [{
              "from_item_attr": x,
              "to_common_attr": x+"_AVG",
              "aggregator": "avg"
            } for x in return_pxtrs+['f1_score']] + [{
              "from_item_attr": x,
              "to_common_attr": x+"_MAX",
              "aggregator": "max"
            } for x in return_pxtrs+['f1_score']]
          )
          .enrich_attr_by_lua(
            import_common_attr=[f"{x}_AVG" for x in return_pxtrs+['f1_score']] + [f"{x}_MAX" for x in return_pxtrs+['f1_score']],
            export_common_attr=[f"{x}_AVG" for x in return_pxtrs+['f1_score']] + [f"{x}_MAX" for x in return_pxtrs+['f1_score']],
            function_for_common="calc",
            lua_script=f"""
                function calc(seq, item_key, reason, score)
                    {EOL.join(f"local {x}_AVG = {x}_AVG*10000.0" for x in return_pxtrs+['f1_score'])}
                    {EOL.join(f"local {x}_MAX = {x}_MAX*10000.0" for x in return_pxtrs+['f1_score'])}
                    return {", ".join([f"{x}_AVG" for x in return_pxtrs+['f1_score']]+[f"{x}_MAX" for x in return_pxtrs+['f1_score']])}
                end
            """
          )
          .self_perf([f"{x}_AVG" for x in return_pxtrs+['f1_score']] + [f"{x}_MAX" for x in return_pxtrs+['f1_score']], namespace, subtag)
          .self_perf(["eval_list_num"], namespace, subtag)
        )

    def eval_call(self):
        return (self
            .if_("item_num > 0")
            .delegate_retrieve(
                kess_service="{{common_eval_kess_name}}",
                send_common_attrs=[{"name": "realshow_photo_id_list", "as":"eval_pos_photo_id_list"}, "tab_id"],
                request_type="eval_request",
                recv_item_attrs=["session_score", "session_photo_id_list"],
                reason=1024,
                timeout_ms=3000,
                request_num=1000
            )
            .log_recall()
            .filter()
            .delegate_enrich(
                kess_service = "grpc_rljListScoreForGen",
                request_type = "eval_request",
                timeout_ms = 1000,
                send_item_attrs=[
                    "session_photo_id_list"
                ],
                send_common_attrs = [
                    "user_info_str", "tab_id", 
                    {"name": "common_eval_kess_name", "as": "eval_kess_name"}
                ],
                recv_item_attrs=[x+"_list" for x in return_pxtrs],
                reason=1024,
                partition_size = 100,
                use_packed_item_attr = True
            )
            .log_reward()
            .enrich_attr_by_lua(
                import_common_attr=["common_eval_kess_list",
                                    "common_eval_kess_list_size",
                                    "common_eval_kess_list_index"],
                export_common_attr=["common_eval_kess_list_index",
                                    "common_eval_kess_name",
                                    "is_keep_call_eval"],
                function_for_common="calculate",
                lua_script="""
                function calculate()
                    local common_eval_kess_list_index = common_eval_kess_list_index or 1
                    common_eval_kess_list_index = common_eval_kess_list_index + 1
                    local is_keep_call_eval = common_eval_kess_list_index <= common_eval_kess_list_size
                    local common_eval_kess_name = ""
                    if (is_keep_call_eval) then
                        common_eval_kess_name = common_eval_kess_list[common_eval_kess_list_index]
                    end
                    return common_eval_kess_list_index,
                           common_eval_kess_name,
                           is_keep_call_eval
                end
                """
            )
            .truncate(size_limit=0)
            .if_("is_keep_call_eval == 0 ")
                .eval_realshow("servershow_photo_id_list")
                .eval_realshow("servershow_consume_photo_id_list")
                .eval_realshow("realshow_photo_id_list")
                .eval_realshow("realshow_consume_photo_id_list")
            .end_if_()
            .end_()
        )

# 第三步，清理现场
class FinishStage(LeafFlow):
    def __init__(self, name):
        LeafFlow.__init__(self, name)

    def finish_clean(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)

call_eval_server = CallEvalServer("call_eval_server", loop_if="is_keep_call_eval", loop_limit=100)
call_eval_server.eval_call()

finish_stage = FinishStage("finish_stage")
finish_stage.finish_clean("finish_eval")

def generate_pipeline():
    runner = OfflineRunner("common-tdm-eval")
    runner.ENABLE_ATTR_CHECK = False
    runner.add_leaf_flows(leaf_flows=[
        read_log,
        call_eval_server,
        finish_stage
    ])
    filepath = os.path.join(current_dir, "dynamic_json_config.json")
    runner.build(output_file = filepath)
    return runner

def generate_debug_pipeline():
    runner = OfflineRunner("common-grm-eval-debug")
    runner.ENABLE_ATTR_CHECK = False
    runner.add_leaf_flows(leaf_flows=[
        read_log,
        call_eval_server,
        finish_stage
    ])
    return runner

if args.run:
    runner = generate_debug_pipeline()
    exe = runner.executor()
    while not exe["MESSAGE_END"]:
        exe.reset()
        exe.run("read_log")
        exe.run("call_eval_server ")
else:
    runner = generate_pipeline()
