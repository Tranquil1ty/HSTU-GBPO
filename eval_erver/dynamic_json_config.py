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
    "vtr_calibration_score", "fuse_score", "new_svr", "sppc_combine",
    "cpr_wtd", "wtd_duration_score_v2", "wtd_v2", "caption_searchpage", "click_live",
    "effective_watch_live_time", "setr3_low_active", "search_pure_cmt_highlight_ctr", "itr",
    "playlet_ctr", "adp_wtd", "hashtag_ctr", "peak_evtr", "search_comment_trending_click",
    "bubble_cr", "hashtag_sppc", "bottom_bar_ctr", "ua_long_term_page_score", "session_play_time",
    "all_evtr", "avtt", "playtime_denoise", "pro_cpr", "total_watch_time", "total_watch_time_wtd",
    "adp_clevtr_pro", "pro_evtr", "watch_live", "wtd_v2_playtime", "cpr_duration_score", "setr2",
    "bubble_sr", "search_page_photo_show", "search_page_photo_click"
]

perf_pxtrs = ["evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", "vtr", "ptr", "lsst", "wtd_v2", "cpr"]
perf_pxtrs += ["session_play_time", "qtr", "dtr", "epstr", "cmef", "cltr", "adp_wtd", "playlet_ctr", "setr2"]
perf_pxtrs += ["f1_score"]

min_pxtrs = ["svr", "htr"]

EOL = '\n'

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
                   group_id="onerec_eval",
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
    .gen_common_attr_by_lua(attr_map={"tab_id_str": "tostring(tab_id)"})
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

    # 第1.1步：解析出 realshow item 的semantic id list
    def pre_eval(self):
        return (self.limit(0, name="clean_all_for_before_eval")
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
            check_point="onerec.eval_length",
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

    # 第1.2步：请求生成模型，并更新下次请求的服务
    def eval_call(self):
        return (self
            .delegate_retrieve(
                kess_service="{{common_eval_kess_name}}",
                send_common_attrs=["eval_pos_photo_id_list", "f1_score_list", "tab_id", {"name": "user_info_str", "as": "user"}] + [
                    "like_photo_id_list", "follow_photo_id_list", "longview_photo_id_list"
                ],
                request_type="default",
                timeout_ms=10000,
                request_num=1000
            )
            .log_debug_info(
                log_tag="hqg_debug.onerec.eval.kess",
                common_attrs=["common_eval_kess_list_size",
                                    "common_eval_kess_list_index",
                                    "common_eval_kess_name",
                                    "is_keep_call_eval"],
                for_debug_request_only=False,
                respect_sample_logging=False
            )
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
        )
    
    def post_eval(self):
        return (self.if_("eval_length ~= nil and eval_length > 0")
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
        .hitrate_perf(namespace="common.leaf", subtag="onerec_hit_rate_eval")
        .end_()
    )

    def reward_eval(self):
        return (self.count_reco_result(save_count_to="item_num")
        .log_debug_info(
                log_tag="hqg_debug.onerec.eval.reward_eval",
                common_attrs=["common_eval_kess_list_size",
                                    "common_eval_kess_list_index",
                                    "common_eval_kess_name",
                                    "is_keep_call_eval", "item_num"],
                for_debug_request_only=False,
                respect_sample_logging=False
        )
        .if_("item_num > 0")
        .copy_item_meta_info(
            save_item_key_to_attr="item_id",
        )
        .get_item_attr_by_distributed_index(
            photo_store_kconf_key="reco.distributedIndex.hotPhotoStoreConfig",
            use_dynamic_photo_store=True,
            attrs=[{ "name": "is_living", "path": "live_photo_info.is_living" },]
        )
        .build_protobuf(
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
            }]
        )
        .enrich_attr_by_lua(
            import_common_attr=["tab_id"],
            export_common_attr=["full_rank_req_type"],
            function_for_common="func",
            lua_script="""
            function func()
                local full_rank_req_type = 'predict_for_gamora'
                if tab_id == 30000 then
                    full_rank_req_type = 'predict_for_nebula'
                end
                return full_rank_req_type
            end
            """
        )
        .delegate_enrich(
            name="delegate_enrich_main_model",
            kess_service="grpc_hqg24q4ModelComboFinal",
            request_type="{{full_rank_req_type}}",
            timeout_ms=10000,
            send_common_attrs = [
                "user_info_str",
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
        .perflog_attr_value(check_point="onerec.eval.reward", item_attrs=["evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", "vtr", "svr", "ptr"],)
        .perf_reward_value(namespace="common.leaf", subtag="onerec_reward_value_eval")
        .end_()
    )

    def perf_reward_value(self, namespace, subtag):

        self.pack_item_attr(
            item_source={ "reco_results": True },
            mappings=[
                {"from_item_attr": x, "to_common_attr": x+"_MAX", "aggregator": "max"} for x in perf_pxtrs
            ] + [
                {"from_item_attr": x, "to_common_attr": x+"_AVG", "aggregator": "avg"} for x in perf_pxtrs + min_pxtrs
            ] + [
                {"from_item_attr": x, "to_common_attr": x+"_MIN", "aggregator": "min"} for x in min_pxtrs
            ]
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
            import_common_attr=[f"{x}_AVG" for x in perf_pxtrs + min_pxtrs],
            export_common_attr=[f"{x}_AVG" for x in perf_pxtrs + min_pxtrs],
            function_for_common="calc",
            lua_script=f"""
            function calc()
                {EOL.join(f"local {x} = math.floor({x}_AVG*10000)" for x in perf_pxtrs + min_pxtrs)}
                return {", ".join(x for x in perf_pxtrs + min_pxtrs)}
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
                extra1="{{common_eval_kess_name}}",
                extra2=x+"_MAX",
                extra3="{{tab_id_str}}"
            )
        
        for x in perf_pxtrs + min_pxtrs:
            self.perflog(
                mode="interval",
                value="{{" + x+"_AVG" + "}}",
                namespace=namespace,
                subtag=subtag,
                extra1="{{common_eval_kess_name}}",
                extra2=x+"_AVG",
                extra3="{{tab_id_str}}"
            )
        
        for x in min_pxtrs:
            self.perflog(
                mode="interval",
                value="{{" + x+"_MIN" + "}}",
                namespace=namespace,
                subtag=subtag,
                extra1="{{common_eval_kess_name}}",
                extra2=x+"_MIN",
                extra3="{{tab_id_str}}"
            )

        # 按精排 排序后的 top6
        self.perf_sorted_reward_value(namespace, subtag)

        return self
    
    def perf_sorted_reward_value(self, namespace, subtag, desc=True):
        (
            self.sort_by("f1_score", desc=desc)
            .copy_item_meta_info(
                save_item_seq_to_attr="item_seq"
            )
            .pack_item_attr(
                item_source={ "reco_results": True },
                mappings=[
                    {"from_item_attr": x, "to_common_attr": x+"_SORTED_TOP6_AVG", "aggregator": "avg"} for x in perf_pxtrs + min_pxtrs
                ],
                target_item = { "item_seq": [0, 1, 2, 3, 4, 5] }
            )
            .enrich_attr_by_lua(
                import_common_attr=[f"{x}_SORTED_TOP6_AVG" for x in perf_pxtrs + min_pxtrs],
                export_common_attr=[f"{x}_SORTED_TOP6_AVG" for x in perf_pxtrs + min_pxtrs],
                function_for_common="calc",
                lua_script=f"""
                function calc()
                    {EOL.join(f"local {x} = math.floor({x}_SORTED_TOP6_AVG*10000)" for x in perf_pxtrs + min_pxtrs)}
                    return {", ".join(x for x in perf_pxtrs + min_pxtrs)}
                end
                """
            )
        )
        for x in (perf_pxtrs + min_pxtrs):
            self.perflog(
                mode="interval",
                value="{{" + x+"_SORTED_TOP6_AVG" + "}}",
                namespace=namespace,
                subtag=subtag,
                extra1="{{common_eval_kess_name}}",
                extra2=x+"_SORTED_TOP6_AVG",
                extra3="{{tab_id_str}}"
            )
        return self
    
    def hitrate_perf(self, namespace, subtag):
        return self \
            .perflog_attr_value(check_point="generative.hit_rate", common_attrs=["hit_rate_1", "hit_rate_2", "hit_rate_3"]) \
            .gen_common_attr_by_lua(attr_map={"tab_id_str": "tostring(tab_id)"}) \
            .perflog(mode="interval", value="{{hit_rate_1}}", namespace=namespace, subtag=subtag, extra1="{{common_eval_kess_name}}", extra2="hit_rate_1", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_2}}", namespace=namespace, subtag=subtag, extra1="{{common_eval_kess_name}}", extra2="hit_rate_2", extra3="{{tab_id_str}}") \
            .perflog(mode="interval", value="{{hit_rate_3}}", namespace=namespace, subtag=subtag, extra1="{{common_eval_kess_name}}", extra2="hit_rate_3", extra3="{{tab_id_str}}") \
            .if_("hit_rate_1 > 0") \
                .perflog(mode="interval", value="{{rank_index_1}}", namespace=namespace, subtag=subtag, extra1="{{common_eval_kess_name}}", extra2="rank_index_1", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_2 > 0") \
                .perflog(mode="interval", value="{{rank_index_2}}", namespace=namespace, subtag=subtag, extra1="{{common_eval_kess_name}}", extra2="rank_index_2", extra3="{{tab_id_str}}") \
            .end_() \
            .if_("hit_rate_3 > 0") \
                .perflog(mode="interval", value="{{rank_index_3}}", namespace=namespace, subtag=subtag, extra1="{{common_eval_kess_name}}", extra2="rank_index_3", extra3="{{tab_id_str}}") \
            .end_()


# 第三步，清理现场
class FinishStage(LeafFlow):
    def __init__(self, name):
        LeafFlow.__init__(self, name)

    def finish_clean(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)

call_eval_server = CallEvalServer("call_eval_server", loop_if="is_keep_call_eval", loop_limit=100)
call_eval_server.pre_eval()
call_eval_server.eval_call()
call_eval_server.reward_eval()
call_eval_server.post_eval()

finish_stage = FinishStage("finish_stage")
finish_stage.finish_clean("finish_eval")

def generate_pipeline():
    runner = OfflineRunner("common-onerec-eval")
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
    runner = OfflineRunner("common-onerec-eval-debug")
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
