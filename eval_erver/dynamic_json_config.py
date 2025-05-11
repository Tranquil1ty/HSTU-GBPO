import os
import argparse
import numpy as np
import sys

from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin 

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../dragon/tools/pypi/"))
sys.path.append('~/dragon/tools/pypi/')
current_dir = os.path.dirname(__file__)
parser = argparse.ArgumentParser()
parser.add_argument("--run", dest="run", default=False, action="store_true")
parser.add_argument("--eval", dest="eval", default=False, action="store_true")
args = parser.parse_args()

class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, GsuApiMixin, EmbedCalcApiMixin, KuibaApiMixin, PDNApiMixin, CofeaApiMixin, KgnnApiMixin):
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
  "wtd_effective_view",

  "formula_one_score"
]

def load_feature_list_sign(filename):
    ret = set()
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            for field in line.strip().split(','):
                parts = field.strip().split('=')
                assert len(parts) == 2, "Unsupported format: " + line.strip()

                if parts[0].strip() != 'class':
                    # ignore unknown field
                    continue

                ret.add(parts[1].strip())
    return list(sorted(ret))

# 第一步，消费无 Lag 采样流量
read_log = (
    DataReaderFlow(name="read_log")
    .fetch_message(output_attr="compressed_batched_samples",
                   group_id="g_r_m_eval",
                #    onerec_eval
                   kafka_topic="kaiworks_retrieval_eval_flow_hb1",
                   begin_time_ms=1746624539000)
    .zstd(
        mode="decompress",
        input_common_attr="compressed_batched_samples",
        output_common_attr="batched_samples",
    )
    .read_cofea_sample(
        sample_from_attr="batched_samples",
        id_feature="photo_id",
        user_id_feature="user_id",
        device_id_feature="device_id",
        extract_common_features=["tab_id", "llsid", "thanos_flag", "user_id", "device_id", "time_ms", "user_hash", "traffic_type", "user_info_str"],
        extract_item_features=["photo_id", "author_id", "duration_ms", "upload_time"] + labels,
    )
    .if_("tab_id ~= 10000 and tab_id ~= 30000")
        .return_()
    .end_()
    .set_attr_value(
        no_overwrite=True,
        item_attrs=[{"name": "formula_one_score", "type": "double", "value": 0.0}]
    )
    .parse_protobuf_from_string(
        is_common_attr=True,
        ttl_seconds=7200,
        input_attr="user_info_str",
        output_attr="user_info",
        class_name="ks::reco::UserInfo",
    )
    .extract_with_ks_sign_feature(
        feature_list=load_feature_list_sign("./feature_list_sign.txt"),
        user_info_attr="user_info",
        common_slots_output="common_user_slots",
        common_parameters_output="common_user_signs",
    )
    .log_debug_info(common_attrs=["common_user_slots", "common_user_signs"], for_debug_request_only=False)
    .copy_user_meta_info(save_result_size_to_attr="item_num")
    .if_("item_num == 0")
        .return_()
    .end_()
    .get_kconf_params(kconf_configs=[{
        "kconf_key": "rinf.rlRunner.OneRecEvalModelList",
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
        target_item={ "realshow": 1 },
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "eval_pos_photo_id_list", }],
    )
    .pack_item_attr(
        target_item={ "realshow": 1 },
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "formula_one_score", "to_common_attr": "f1_score_list", }],
    )
    .pack_item_attr(
        target_item={ "like": 1 },
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "like_photo_id_list", }],
    )
    .pack_item_attr(
        target_item={ "follow": 1 },
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "follow_photo_id_list", }],
    )
    .pack_item_attr(
        target_item={ "long_view": 1 },
        item_source={ "reco_results": True, },
        mappings=[{ "from_item_attr": "photo_id", "to_common_attr": "longview_photo_id_list", }],
    )
)

# 第二步, 构造评估样本 & 请求下游进行评估
class CallEvalServer(LeafFlow):
    def __init__(self, name, loop_if, loop_limit):
        LeafFlow.__init__(self, name=name, loop_if=loop_if, loop_limit=loop_limit)

    def eval_call(self):
        return (self
            .if_("item_num > 0")
            .delegate_retrieve(
                kess_service="{{common_eval_kess_name}}",
                send_common_attrs=["eval_pos_photo_id_list", "f1_score_list", "tab_id", {"name": "user_info_str", "as": "user"}] + [
                    "like_photo_id_list", "follow_photo_id_list", "longview_photo_id_list"
                ],
                request_type="eval_request",
                timeout_ms=10000,
                request_num=1000,
                save_result_to_common_attr="eval_retr_photo_id_list"
            ) \
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
