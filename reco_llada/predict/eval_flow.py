from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.common.common_api_mixin import CommonApiMixin
from dragonfly.common_leaf_dsl import LeafFlow

from dragonfly.matx.dragonfly_context import DragonflyContext
# from infer_llada import kess_name
EOL = "\n"

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

class EvalFlow(LeafFlow,OfflineApiMixin, CommonApiMixin):

    def perf_reward_value(self, namespace, subtag, service_name):
        perf_pxtrs = ["evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", "vtr", "svr", "ptr", "htr", "lsst", "wtd_v2", "cpr"]
        self.pack_item_attr(
            item_source={ "reco_results": True },
            mappings=[
                {"from_item_attr": x, "to_common_attr": x+"_MAX", "aggregator": "max"} for x in perf_pxtrs
            ] + [
                {"from_item_attr": x, "to_common_attr": x+"_AVG", "aggregator": "avg"} for x in perf_pxtrs
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
        return self

    def eval_pointwise_reward(self, kess_name):
        return (
            self.count_reco_result(save_count_to="item_num")
            .if_("item_num > 0")
                .gen_common_attr_by_lua(attr_map={"tab_id_str": "tostring(tab_id)"})
                .copy_item_meta_info(
                    save_item_key_to_attr="item_id",
                )
                .get_item_attr_by_distributed_index(
                    photo_store_kconf_key="reco.distributedIndex.recoExploreFastPhotoStoreConfigNuma",
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
                    kess_service="{{reward_service_kess_name}}",
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
                .perflog_attr_value(check_point="default.reward", item_attrs=["evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", "vtr", "svr", "ptr"],)
                .perf_reward_value(namespace="common.leaf", subtag="reward_value_pxtr", service_name=kess_name)
            .end_()
        )
