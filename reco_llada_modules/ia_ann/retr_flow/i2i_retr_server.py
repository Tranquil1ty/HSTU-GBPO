#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys

current_dir = os.path.dirname(__file__)
sys.path.append(
    os.path.join(
        current_dir,
        "../../../kbuild_workspace/ks/common_reco/ann_retrieve/dragonfly/",
    )
)
sys.path.append(
    os.path.join(current_dir, "../../../kbuild_workspace/dragon/tools/pypi/")
)

from dragonfly.common_leaf_dsl import LeafFlow, LeafService, IndexSource
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.oversea.oversea_api_mixin import OverseaApiMixin
from dragonfly.ext.nr.nr_api_mixin import NrApiMixin
kconf_key = "reco.start_auto_param.i2i_ia_emb_server_config"
kconf_config = {
    # 默认的kconf配置值
    # colossus trigger
    "colossus_filter_ev": 0,
    "colossus_filter_lv": 1,
    "colossus_trigger_max_num": 500,
    "pdn_trigger_number": 250,
    "pdn_trigger_type": 0, #指定 pdn trigger 选取类型，支持 0（按 play_time），1（按完播率），2（随机），3（按概率选前三种）
    "pdn_trigger_random_rates": [1.0, 0.0, 0.0],
    "pdn_tag_number": 50,
    "pdn_trigger_num_per_tag": 5,
    "swing_trigger_number": 50,
    "swing_trigger_sample_range": 500,
    "swing_trigger_sample_rate": 0.05,
    "swing_add_positive_feedback": 0,
    "ltv_trigger_number": 50,
    "ltv_trigger_type": 2,
    "ltv_single_tag_max_num": 10,
    "ltv_trigger_past_time_range": 500,
    "ltv_trigger_future_time_range": 100,
    "ltv_tag_trigger_threshold": 10.0,
    "ltv_aid_trigger_threshold": 10.0,
    "interact_trigger_number": 30,
    "interact_trigger_sample_dist": "history",
    "interact_trigger_sample_rates": [1.0, 0.0, 0.0, 0.0, 0.0],

    "enable_missing_memory_trigger": 1,
    "cluster_missing_day_num": 7,
    "cluster_valid_view_num": 10,
    "author_missing_day_num": 7,
    "author_valid_view_num": 5,
    "author_cluster_valid_view_num": 20,
    "author_cluster_missing_day_num": 3,
    # select photo
    "top_cluster_number": 10,
    "photo_num_per_cluster": 5,
    "top_author_number": 10,
    "photo_num_per_author": 5,
    "top_author_cluster_number": 10,
    "photo_num_per_author_cluster": 5,

    "trigger_type": "short",

    "i2i_trigger_max_num": 160,
    "i2i_kgnn_service_name": "grpc_kgnn_i2i_ia_index-I2I",
    "i2i_kgnn_shard_num": 4, 
    "i2i_kgnn_topk": 20, 
    "i2i_kgnn_timeout": 50,
    "i2i_kgnn_min_weight": 0.6,

    # 概率刷新
    "rewrite_trigger_rate": 0.5,

    # 后处理
    "i2i_reduent_score_thresh": 0.97,
    # low pass 多样性过滤
    "enable_low_pass_filter": 1,
    "debug_low_pass": 0,
    "retr_limit_aid_num": 3,
    "retr_limit_hetu_cluster_num": 100,
    "retr_limit_hetu_l1_num": 100,
    "retr_limit_hetu_l2_num": 20,

    # 过滤短视频
    "enable_filter_short_photo_by_threshold": 0, 
    "enable_filter_short_photo_by_deciles": 0, 
    "filter_short_photo__duration_ms_threshold": 15000,
    "filter_short_photo__duration_deciles": 3,
    # post rank 排序
    "enable_post_rank": 0,
    "enable_post_rank_local_cache": 0,
    "post_rank__input_num": 2000,
    "post_rank__pxtr_timeout_ms": 20,
    "post_rank__pxtr_label": ["ctr", "vtr", "pc", "lvr", "u2a_ctr", "abs_lvtr", "eps"],
    "post_rank__pxtr_weight": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "post_rank__min_pxtr": 0.0,
    "post_rank__result_num": 0,
    "enable_post_rank_weighted_pxtr": 0,
    "post_rank__duration_ms_norm_threshold": 15000,

    "i2i_retr_num": 2000
}
 
ab_config_map = {
    "biz_name": "KUAISHOU_APPS",
    "ab_params": [
       ("ia_retr_kconf_key_nebula", "reco.start_auto_param.i2i_ia_emb_server_config"),
       ("ia_retr_kconf_key_gamora", "reco.start_auto_param.i2i_ia_emb_server_config"),
    ]
}

missed_trigger_btq_config = {
    "queue_prefix":'missed_trigger_pid'
}

class I2IRetrFlow(
    LeafFlow, MioApiMixin, GsuApiMixin, PDNApiMixin, KgnnApiMixin, EmbedCalcApiMixin, RetrievalApiMixin, OverseaApiMixin, NrApiMixin
):
    def _extract_kconf_config(self):
        self.get_kconf_params( #从 kconf 系统中获取配置值并作为 CommonAttr 或 ItemAttr 填入 Context 中
            kconf_configs=[
                {
                    "kconf_key": "{{kconf_key}}",
                    "export_common_attr": k,
                    "json_path": k,
                    "default_value": v,
                } for k, v in kconf_config.items()
            ]
        )
        return self

    def _pre_process(self, **kwargs):
        return (
            self.copy_user_meta_info(
                save_request_num_to_attr="request_num",
                save_user_id_to_attr="user_id",
                save_request_type_to_attr="request_type",
            )
            .get_abtest_params(
                **ab_config_map,
            )
            .if_("request_type == 'nebula'") 
                .gen_common_attr_by_lua(
                    attr_map={
                        "kconf_key": "ia_retr_kconf_key_nebula",
                    }
                )
            .else_()
                .gen_common_attr_by_lua(
                    attr_map={
                        "kconf_key": "ia_retr_kconf_key_gamora",
                    }
                )
            .end_()
            # 自定义 kconf key 覆盖
            .if_("custom_kconf_key ~= nil")
                .gen_common_attr_by_lua(
                    attr_map={
                        "kconf_key": "custom_kconf_key",
                    }
                )
            .end_()
            ._extract_kconf_config() 
            .gen_common_attr_by_lua(
                attr_map={
                    "limit_num": "math.min(request_num and request_num or 2000, i2i_retr_num)", 
                }
            )

            .if_("user == nil")
                .return_(2, "no user info")
            .end_if_()
            .parse_protobuf_from_string( #从protobuf中读取信息
                input_attr="user",
                output_attr="user_info",
                class_name="ks::reco::UserInfo",
            )
            .enrich_with_protobuf(
                from_extra_var="user_info",
                is_common_attr=True,
                attrs=[
                    dict(path="browsed_photo_ids", name="browsed_photo_ids"),
                    dict(
                        path="user_profile_v1.real_show_list.photo_id",
                        name="userprofile_realshow_pid",
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
                    dict(
                        path="user_profile_v1.video_playing_stat.video_duration",
                        name="video_playing_stat_video_duration",
                    ),
                    dict(
                        path="user_profile_v1.video_playing_stat.playing_time",
                        name="video_playing_stat_play_time",
                    ),
                    dict(
                        path="user_profile_v1.video_playing_stat.photo_id",
                        name="video_playing_stat_photo_id",
                    ),
                ],
            )
            .pack_common_attr(
                input_common_attrs=[
                    "browsed_photo_ids",
                    "nearby_browsed_photo_ids",
                    "slide_browsed_photo_ids",
                    "click_list",
                    "like_list",
                    "userprofile_realshow_pid",
                    "video_playing_stat_photo_id",
                ],
                output_common_attr="browsed_pids",
                deduplicate=True,
            )

        )
    
    def _fetch_trigger(self):
        from fetch_trigger_flow import fetch_trigger_flow 
        self.enrich_by_sub_flow(
            sub_flow=fetch_trigger_flow,
            deepcopy=False,
            merge_common_attrs=[
                "trigger_list",
                "trigger_num",
                "pdn_item_trigger",
                "colossus_trigger",
                "swing_item_trigger",
                "ltv_item_trigger",
                "interact_item_trigger",
                "user_profile_item_trigger",
            ]
        )
        return self

    def _user_infer(self):
        from nebula_mc_user_tower_infer_flow import nebula_mc_user_tower_infer_flow
        self.enrich_by_sub_flow(
            sub_flow=nebula_mc_user_tower_infer_flow,
            deepcopy=False,
            merge_common_attrs=[
                "user_top_layer",
            ]
        )
        return self
        
    def post_rank(self):
        tower_labels = [
            "ctr", "lvr", "svr", "ltr", "wtr", "ftr", "eps", "cmtr", 
            "ces", "live", "vtr", "down", "clkcmt", "swptr", "htr", 
            "pc","product_shoot_score","mtmpc_ctr","mtmpc_after", "lvtr2", "u2a_ctr", 
            "fr_mix", "abs_lvtr", "pair_evtr", "dcm_realshow1", "dcm_realshow2",
        ]
        return (
            self
            .count_reco_result(save_count_to="candiate_item_num")
            .if_("user_top_layer ~= nil and candiate_item_num > post_rank__result_num")
                .truncate(size_limit="{{post_rank__input_num}}")
                .fetch_tower_remote_pxtr( # 向远端请求预估多个label (or target) 的多个 user * item 的pxtr
                    predict_labels=tower_labels,
                    common_embedding_len=136 * len(tower_labels),
                    user_embedding_attr="user_top_layer",
                    use_item_key_as_embed_key=True,
                    output_type=2,
                    pxtr_kess_service="grpc_nebulaTowerFullLinkPxtr",
                    pxtr_shards=7,
                    pxtr_subreq_num_in_shard=2,
                    pxtr_timeout_ms='{{post_rank__pxtr_timeout_ms}}',
                    pxtr_req_type='unifea_tower_predict_pxtr',
                    pxtr_req_common_embedding_attr='req_common_embedding',
                    pxtr_req_tower_caller_attr='tower_caller',
                    pxtr_return_value_attr='return_pxtr_value',
                )
                # 加权分: pxtr
                .fetch_post_rank_pxtr(
                    all_pxtr_labels=tower_labels,
                    pxtr_labels='{{post_rank__pxtr_label}}',
                    # pxtr_labels=["ctr", "vtr", "pc", "lvr", "u2a_ctr", "abs_lvtr", "eps"],
                    pxtr_weights='{{post_rank__pxtr_weight}}',
                    pxtr_attr="pxtr",
                )

                .perflog_attr_value(
                    check_point="{{return kconf_key .. '.post_rank.before'}}",
                    item_attrs=[
                        "ctr", "vtr", "pc", "lvr", "svr", "abs_lvtr", "u2a_ctr", "eps", "ltr", "ftr",
                        "pxtr", "duration_ms"
                    ],
                )
                # 按 pxtr 排序
                .filter_by_attr(
                    attr_name="pxtr", 
                    remove_if="<=", 
                    compare_to='{{post_rank__min_pxtr}}', 
                    remove_if_attr_missing=False
                )
                .enrich_attr_by_lua(
                    import_item_attr=["duration_ms", "pxtr"],
                    import_common_attr=["enable_post_rank_weighted_pxtr", "post_rank__duration_ms_norm_threshold"],
                    export_item_attr=["duration_norm", "pxtr_label"],
                    function_for_item="cal",
                    lua_script="""
                    function cal()
                        local duration_norm = duration_ms / math.max(post_rank__duration_ms_norm_threshold, duration_ms)
                        local pxtr_label = 0
                        if (enable_post_rank_weighted_pxtr == 1) then
                            pxtr_label = pxtr * duration_norm
                        else
                            pxtr_label = pxtr
                        end
                        return duration_norm, pxtr_label
                    end 
                    """
                ) 
                .sort_by(attr="pxtr_label")
                .truncate_by_attr(attr_name="pxtr_label", size_limit="{{post_rank__result_num}}")
                .perflog_attr_value(
                    check_point="{{return kconf_key .. '.post_rank.after'}}",
                    item_attrs=[
                        "ctr", "vtr", "pc", "lvr", "svr", "abs_lvtr", "u2a_ctr", "eps", "ltr", "ftr",
                        "pxtr", "duration_ms", "pxtr_label"
                    ],
                )
            .end_()
        )
        

    def _retrieve(self, **kwargs):
        return (
            self
            # trigger 选择
            ._fetch_trigger()
            ._user_infer()
            .limit(0)
            .if_("trigger_list ~= nil and trigger_num > 0")
                .shuffle_list_attr(common_attr="trigger_list")
                .truncate(size_limit="{{i2i_trigger_max_num}}", item_list_from_attr="trigger_list") 
                ._get_pid_from_kgnn()
                ._post_process_strategy()
                .shuffle()
                .if_("enable_post_rank ~= 0")
                    .post_rank() 
                .end_()
                .count_reco_result(save_count_to="after_post_rank_item_num")
                ._diversity_filter()
                .count_reco_result(save_count_to="diversity_filter_item_num")
                .truncate(size_limit="{{limit_num}}")
                .count_reco_result(save_count_to="result_item_num")
                .perflog_attr_value(
                    check_point="{{return kconf_key .. '.retr_stats'}}",
                    common_attrs=[
                        "trigger_num", # 获得的trigger数量
                        "i2i_trigger_max_num", # 限制的最大trigger数量
                        "kgnn_res_item_num", # 原始的kgnn返回数量
                        "valid_item_num", # 有效的kgnn返回数量
                        "deduplicate_valid_item_num", # 去重后的有效的kgnn返回数量
                        "index_filter_item_num", # 索引过滤后
                        "reduent_filter_item_num", # 冗余过滤后
                        "short_filter_item_num", # 过滤短视频后 
                        "after_post_rank_item_num", 
                        "diversity_filter_item_num",
                        "i2i_kgnn_topk",
                        "request_num",
                        "i2i_retr_num",
                        "limit_num",
                        "result_item_num",  # 最终的kgnn返回数量
                    ],
                )
            .end_()
        )
    
    def _get_pid_from_kgnn(self):
        self.fetch_kgnn_neighbors(
            kess_service="{{i2i_kgnn_service_name}}",
            id_from_common_attr=f"trigger_list",
            save_neighbors_to = "retr_kgnn_pid",
            save_weight_to= "retr_kgnn_weight",
            relation_name = "I2I",
            shard_num ="{{i2i_kgnn_shard_num}}",
            sample_num ="{{i2i_kgnn_topk}}",
            timeout_ms="{{i2i_kgnn_timeout}}",
            sample_type = 'topn',
            padding_type = 'zero',
            min_weight = "{{i2i_kgnn_min_weight}}",
            request_num=10000,
        )
        self.gen_common_attr_by_lua(
            attr_map={
                "kgnn_res_item_num": "#(retr_kgnn_pid or {})",
            }
        )

        # 分配 每个kgnn召回结果的 trigger id
        self \
        .retrieve_by_common_attr(attr="retr_kgnn_pid", reason=1) \
        .dispatch_common_attr(
            from_common_attr="retr_kgnn_weight", 
            to_item_attr = "score"
        )

        self.send_missed_trigger_with_btq()
        self.enrich_attr_by_lua(
            import_common_attr=["trigger_list", "i2i_kgnn_topk"],
            export_common_attr=["repeat_trigger_list"],
            function_for_common="repeatElements",
            lua_script="""
            function repeatElements()
                --- 复制 trigger_list 各 i2i_kgnn_topk 遍
                local repeat_trigger_list = {}
                for _, value in ipairs(trigger_list) do
                    for i = 1, i2i_kgnn_topk do
                        table.insert(repeat_trigger_list, value)
                    end
                end
                return repeat_trigger_list
            end 
            """
        ) \
        .dispatch_common_attr(
            from_common_attr="repeat_trigger_list", 
            to_item_attr = "from_trigger_id"
        ) \
        .copy_item_meta_info(save_item_id_to_attr="pid") \
        .filter_by_attr( # 过滤无效视频
            attr_name="pid",
            remove_if="<=",
            compare_to=0,
            remove_if_attr_missing=True,
        )
        self.count_reco_result(
            save_count_to="valid_item_num" # 有效的召回结果数量
        )
        self.deduplicate()
        self.count_reco_result(
            save_count_to="deduplicate_valid_item_num" # 有效的去重召回结果数量
        )

        return self
    
    def _post_process_strategy(self):
        # 对于从 kgnn 中检索得到的item，过滤不在索引池中或vv<100的item
        return (
            self
            .filter_by_browse_set()
            .filter_by_common_attr(common_attr=[
                "browsed_pids", 
                "trigger_list", 
                "pdn_item_trigger", 
                "user_profile_item_trigger"
            ])
            .get_item_attr_by_distributed_flat_index( #获取PhotoInfo，只支持主站精选
                photo_store_kconf_key = "reco.distributedIndex.nrPhotoInfoCommonIndex",
                use_dynamic_photo_store = True,
                attrs = [
                    "photo_id",
                    "duration_ms",
                    "author__id",
                    "thanos_stats__real_show_count",
                    "nebula_stats__real_show_count",
                    "hetu_tag_level_info__hetu_level_one",
                    "hetu_tag_level_info__hetu_level_two",
                    "hetu_tag_level_info__hetu_cluster_id",
                ]
            )
            .enrich_attr_by_lua( 
                import_item_attr=[
                    "thanos_stats__real_show_count",
                    "nebula_stats__real_show_count",
                    "photo_id",
                    "duration_ms",
                ],
                export_item_attr=[
                    "is_remove",
                ],
                function_for_item="calc",
                lua_script="""
                function calc()
                    local is_remove = 0;
                    local n_vv = nebula_stats__real_show_count or 0;
                    local g_vv = thanos_stats__real_show_count or 0;
                    local total_vv = n_vv + g_vv;
                    local duration = duration_ms or -1;
                    local pid = photo_id or -1;
                    if ((pid < 0) or (duration < 0) or (total_vv < 100)) then
                        is_remove = 1;
                    end 
                    return is_remove;
                end
                """
            )
            .filter_by_attr(
                attr_name="is_remove",
                remove_if="==",
                compare_to=1,
                remove_if_attr_missing=True,
            )
            .count_reco_result(
                save_count_to="index_filter_item_num",
            )
            # 高度雷同的内容过滤，暂用 score
            .filter_by_attr(
                attr_name="score",
                remove_if=">=",
                compare_to="{{i2i_reduent_score_thresh}}",
                remove_if_attr_missing=True,
            )
            .count_reco_result(
                save_count_to="reduent_filter_item_num",
            )
            .filter_short_photos()
        )
        
    def filter_short_photos(self):
        return (
            self 
            .perflog_attr_value(
                    check_point="{{return kconf_key .. '.filter_short.before'}}",
                    item_attrs=["duration_ms"],
            )
            .if_("enable_filter_short_photo_by_threshold == 1")
                .filter_by_attr(
                    attr_name="duration_ms",
                    remove_if="<=",
                    compare_to="{{filter_short_photo__duration_ms_threshold}}",
                    remove_if_attr_missing=True,
                )
            .else_if_("enable_filter_short_photo_by_deciles == 1")
                .pack_item_attr(
                    item_source = {
                        "reco_results": True,
                    },
                    mappings = [{
                        "from_item_attr": "duration_ms",
                        "to_common_attr": "duration_ms_list",
                    }]
                )
                .nr_calc_fractile(
                    common_attrs=["duration_ms_list"],
                    fractile_num=10, 
                    output_fractile_prefix="fractile_of_"
                )
                .log_debug_info(
                    for_debug_request_only=True,
                    common_attrs = ['duration_ms_list', "fractile_of_duration_ms_list"]
                )
                .enrich_attr_by_lua(
                    import_common_attr = ["fractile_of_duration_ms_list", "filter_short_photo__duration_deciles"],
                    function_for_common = "calculate",
                    export_common_attr = ["duration_ms_threshold"],
                    lua_script = """
                        function calculate()
                        return fractile_of_duration_ms_list[filter_short_photo__duration_deciles]
                        end
                    """
                )
                .filter_by_attr(
                    attr_name="duration_ms",
                    remove_if="<=",
                    compare_to="{{duration_ms_threshold}}",
                    remove_if_attr_missing=True,
                )
            .end_()
            .count_reco_result(
                    save_count_to="short_filter_item_num",
                )
            .log_debug_info(
                for_debug_request_only=True,
                common_attrs = ['reduent_filter_item_num', "short_filter_item_num"]
            )
            .perflog_attr_value(
                    check_point="{{return kconf_key .. '.filter_short.after'}}",
                    item_attrs=["duration_ms"],
            )
        )
    
    def _diversity_filter(self):
        # Aid /Hetu 类目限制过滤
        return (
            self.if_("enable_low_pass_filter ~= 0")
                .pack_item_attr(
                    item_source = {
                        "reco_results": True,
                    },
                    mappings = [
                        {"from_item_attr": "author__id", "to_common_attr": "author_id_list", "dedup_to_common_attr": True},
                        {"from_item_attr": "hetu_tag_level_info__hetu_cluster_id", "to_common_attr": "hetu_cluster_list", "dedup_to_common_attr": True},          
                        {"from_item_attr": "hetu_tag_level_info__hetu_level_one", "to_common_attr": "hetu_level_one_list", "dedup_to_common_attr": True},          
                        {"from_item_attr": "hetu_tag_level_info__hetu_level_two", "to_common_attr": "hetu_level_two_list", "dedup_to_common_attr": True},          
                    ]
                )
                .gen_common_attr_by_lua(
                    attr_map={
                        "author_id_list_uniq_cnt": "#(author_id_list or {})",
                        "hetu_cluster_list_uniq_cnt": "#(hetu_cluster_list or {})",
                        "hetu_level_one_list_uniq_cnt": "#(hetu_level_one_list or {})",
                        "hetu_level_two_list_uniq_cnt": "#(hetu_level_two_list or {})",
                    }  
                )
                .retrieval_low_pass_diversity(
                    is_debug="{{debug_low_pass}}",
                    config=[
                        {"attr": "author__id", "limit": "{{retr_limit_aid_num}}"},
                        {"attr": "hetu_tag_level_info__hetu_cluster_id", "limit": "{{retr_limit_hetu_cluster_num}}"},
                        {"attr": "hetu_tag_level_info__hetu_level_one", "limit": "{{retr_limit_hetu_l1_num}}"},
                        {"attr": "hetu_tag_level_info__hetu_level_two", "limit": "{{retr_limit_hetu_l2_num}}"},
                    ]
                )
                .perflog_attr_value(
                    check_point="{{return kconf_key .. '.low_pass_diversity'}}",
                    common_attrs=[
                        "author_id_list_uniq_cnt",
                        "hetu_cluster_list_uniq_cnt",
                        "hetu_level_one_list_uniq_cnt",
                        "hetu_level_two_list_uniq_cnt",
                        "retr_limit_aid_num",
                        "retr_limit_hetu_cluster_num",
                        "retr_limit_hetu_l1_num",
                        "retr_limit_hetu_l2_num",
                    ],
                )
            .end_()
        )

    def send_missed_trigger_with_btq(self):
        """需要重写刷 kgnn 的情况
        1) 不命中的 miss trigger
        2) 检索的分数较低/检索的结果较少 # TODO: 后续考虑
        3) 按概率重刷
        """
        self.enrich_attr_by_lua(
            import_common_attr=["retr_kgnn_pid", "i2i_kgnn_topk", "trigger_list", "rewrite_trigger_rate"],
            export_common_attr=["missed_trigger_list"],
            function_for_common="get_missed_trigger",
            lua_script="""
            function get_missed_trigger()
                --- 获得 miss 的 trigger pid
                local trigger_num = #(trigger_list or {})
                local missed_trigger_list = {}
                for trigger_index = 1, trigger_num do
                    local all_zero = true
                    local retr_cnt = 0
                    for retr_index = 0, i2i_kgnn_topk-1 do
                        if retr_kgnn_pid[(trigger_index - 1) * i2i_kgnn_topk + retr_index + 1] ~= 0 then
                          all_zero = false
                          break
                        end
                    end
                    if (all_zero or (math.random() <= rewrite_trigger_rate)) then
                        table.insert(missed_trigger_list, trigger_list[trigger_index])
                    end

                end
                return missed_trigger_list
            end 
            """
        )
        self.gen_common_attr_by_lua(
            attr_map = {
              "missed_trigger_num": "#(missed_trigger_list or {})"
            }
        )
        self.if_("missed_trigger_num > 0")
        # send btq of missed_trigge_pid
        self.enrich_attr_by_lua(
            debug_info = True,
            import_common_attr=["missed_trigger_list", "missed_trigger_num"],
            export_common_attr=["missed_trigger_list_string"],
            function_for_common="intArrayToString",
            lua_script="""
            function intArrayToString()
                local missed_trigger_list_string = ''
                if (missed_trigger_num ~= 0) then
                    missed_trigger_list_string = table.concat(missed_trigger_list,',')
                end
                return missed_trigger_list_string
            end
            """
        )
        self.send_with_btq(
            common_attr=f"missed_trigger_list_string",
            queue_name=f"{missed_trigger_btq_config['queue_prefix']}",
        ) 
        self.end_()
        self.perflog_attr_value(
            check_point="{{return kconf_key .. '.send_missed_trigger_with_btq'}}",
            common_attrs=["missed_trigger_num"]
        )
        return self

  


# 正常召回 u2i2i
flow = I2IRetrFlow(name="i2i_flow")._pre_process()._retrieve()
# 暂不做策略过滤，纯取 kgnn 对比
flow_demo = I2IRetrFlow(name="flow_i2i_demo")._pre_process()._get_pid_from_kgnn()

service = LeafService(
    kess_name="grpc_slideI2ICenterServer-i2i-bigcode-emb-retr-server",
    common_attrs_from_request=["user", "custom_kconf_key", "trigger_list"],
)
service.return_item_attrs(["from_trigger_id", "score"])

LeafService.CHECK_UNUSED_ATTR = False
service.add_leaf_flows(request_type="default", leaf_flows=[flow], as_default=True)
service.add_leaf_flows(request_type="i2i_retr_flow_demo", leaf_flows=[flow_demo])

service.build(__file__.replace(".py", ".json"))
service.draw(dag_folder="./dag_folder")