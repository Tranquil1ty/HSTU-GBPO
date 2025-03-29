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

kconf_key = "rinf.rlRunner.i2i_ia_emb_server_config_ann"
kconf_config = {
    # colossus trigger
    "colossus_filter_ev": 0,
    "colossus_filter_lv": 1,
    "colossus_trigger_max_num": 500,
    "pdn_trigger_number": 250,
    "pdn_trigger_type": 0,
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

    "enable_short_term_trigger": 0,

    "i2i_trigger_max_num": 160,
    "i2i_ann_service_name": "kws-kuaishou-retrieve-com-retr-mmu-e2e-i2i-index-ann-v0", 
    "i2i_ann_topk": 20, 
    "i2i_ann_timeout": 50,

    # 后处理
    # post rank 排序
    "enable_post_rank": 0,
    "enable_post_rank_local_cache": 0,
    "post_rank__input_num": 2000,
    "post_rank__pxtr_timeout_ms": 20,
    "post_rank__pxtr_label": ["ctr", "vtr", "pc", "lvr", "u2a_ctr", "abs_lvtr", "eps"],
    "post_rank__pxtr_weight": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "post_rank__min_pxtr": 0.0,
    "post_rank__result_num": 0,

    "i2i_retr_num": 2000
}
 
class I2IRetrFlow(
    LeafFlow, MioApiMixin, GsuApiMixin, PDNApiMixin, KgnnApiMixin, EmbedCalcApiMixin
):
    def _extract_kconf_config(self):
        return self.get_kconf_params(
            kconf_configs=[
                {
                    "kconf_key": kconf_key,
                    "export_common_attr": k,
                    "json_path": k,
                    "default_value": v,
                } for k, v in kconf_config.items()
            ]
        )

    def _pre_process(self, **kwargs):
        return (
            self.copy_user_meta_info(
                save_request_num_to_attr="request_num",
                save_user_id_to_attr="user_id",
            )
            ._extract_kconf_config() 
            .gen_common_attr_by_lua(
                attr_map={
                    "limit_num": "math.min(request_num and request_num or 2000, i2i_retr_num)",
                }
            )
            .if_("user == nil")
                .return_(2, "no user info")
            # .fetch_user_info(
            #     kess_service="grpc_recoUserProfileNewTest_recoUserProfileRpcService",
            #     biz_name="Default",
            #     product=17,
            #     save_to_common_attr="user",
            # )
            .end_if_()
            .parse_protobuf_from_string(
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
            .if_("user_top_layer ~= nil and candiate_item_num > limit_num")
                .truncate(size_limit="{{post_rank__input_num}}")
                .fetch_tower_remote_pxtr(
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
                    check_point=f"{self.name}.post_rank.before",
                    item_attrs=[
                        "ctr", "vtr", "pc", "lvr", "svr", "abs_lvtr", "u2a_ctr", "eps", "ltr", "ftr",
                        "pxtr",
                    ],
                )
                # 按 pxtr 排序
                .filter_by_attr(
                    attr_name="pxtr", 
                    remove_if="<=", 
                    compare_to='{{post_rank__min_pxtr}}', 
                    remove_if_attr_missing=False
                )
                .sort_by(attr="pxtr")
                .truncate_by_attr(attr_name="pxtr", size_limit="{{post_rank__result_num}}")
                .perflog_attr_value(
                    check_point=f"{self.name}.post_rank.after",
                    item_attrs=[
                        "ctr", "vtr", "pc", "lvr", "svr", "abs_lvtr", "u2a_ctr", "eps", "ltr", "ftr",
                        "pxtr",
                    ],
                )
            .end_()
        )
        
    def _select_trigger(self, **kwargs):
        return (
            self
            # long term trigger
            .colossus(
                service_name="grpc_colossusRecoSimItemV3",
                client_type="common_item_client",
                output_attr="colossus_resp",
                parse_to_pb=False,
            )
            .enrich_item_trigger(
                name="item_trigger_1w",
                colossus_service_name="grpc_colossusRecoSimItemV3",
                colossus_resp_attr="colossus_resp",
                colossus_output_type="common_item",
                use_cluster_id=False,
                filter_ev="{{colossus_filter_ev}}",
                filter_lv="{{colossus_filter_lv}}",
                filter_future_ts=False,
                export_all_trigger_list="colossus_trigger",
                export_pdn_trigger_list="pdn_item_trigger",
                export_swing_trigger_list="swing_item_trigger",
                export_ltv_trigger_list="ltv_item_trigger",
                export_interact_trigger_list="interact_item_trigger",
                export_time_interest_trigger_list="time_interest_trigger",
                all_trigger_number="{{colossus_trigger_max_num}}",
                pdn_trigger_number="{{pdn_trigger_number}}",
                pdn_trigger_type="{{pdn_trigger_type}}",
                pdn_trigger_random_rates="{{pdn_trigger_random_rates}}",
                pdn_top_tag_number="{{pdn_tag_number}}",
                pdn_trigger_num_per_tag="{{pdn_trigger_num_per_tag}}",
                swing_trigger_number="{{swing_trigger_number}}",
                swing_trigger_sample_range="{{swing_trigger_sample_range}}",
                swing_trigger_sample_rate="{{swing_trigger_sample_rate}}",
                swing_add_positive_feedback="{{swing_add_positive_feedback}}",
                ltv_trigger_number="{{ltv_trigger_number}}",
                ltv_trigger_type="{{ltv_trigger_type}}",
                ltv_enable_diversity=1,
                ltv_single_tag_max_num="{{ltv_single_tag_max_num}}",
                ltv_trigger_past_time_range="{{ltv_trigger_past_time_range}}",
                ltv_trigger_future_time_range="{{ltv_trigger_future_time_range}}",
                ltv_tag_trigger_threshold="{{ltv_tag_trigger_threshold}}",
                ltv_aid_trigger_threshold="{{ltv_aid_trigger_threshold}}",
                interact_trigger_number="{{interact_trigger_number}}",
                interact_trigger_sample_dist="{{interact_trigger_sample_dist}}",
                interact_trigger_sample_rates="{{interact_trigger_sample_rates}}",
                export_latest_timestamp="colossus1w_latest_timestamp",
                export_oldest_timestamp="colossus1w_oldest_timestamp",
            ) 
            
            # short term trigger
            .enrich_attr_by_lua(
                import_common_attr=[
                    "video_playing_stat_play_time",
                    "video_playing_stat_photo_id",
                    "video_playing_stat_video_duration",
                ],
                export_common_attr=[
                    "user_profile_item_trigger",
                    "user_profile_trigger_num",
                ],
                function_for_common="select_trigger",
                lua_script="""
                function is_long_view(duration_ms, playing_time)
                    local playing_time = playing_time or 0;
                    local duration_ms = duration_ms or 0;
                    local long_view = 0
                    
                    if duration_ms <= 3000 then
                        long_view = (playing_time >= 18000)
                    elseif duration_ms > 36000 then
                        long_view = (playing_time > 36000)
                    else
                        long_view = (playing_time >= (duration_ms * 28 + 180000) / 33)
                    end
                    return long_view
                end

                function is_effective_view(duration_ms, playing_time)
                    local playing_time = playing_time or 0;
                    local duration_ms = duration_ms or 0;
                    return (playing_time >= 7000 and playing_time >= duration_ms) or playing_time >= 18000;
                end

                function select_trigger()
                    local trigger_list = {};
                    local video_playing_stat_photo_id = video_playing_stat_photo_id or {}

                    if #video_playing_stat_photo_id> 0 then
                        for iter=1, #video_playing_stat_photo_id do
                            if (is_long_view(video_playing_stat_video_duration[iter], video_playing_stat_play_time[iter])) then
                                table.insert(trigger_list, video_playing_stat_photo_id[iter]);
                            end
                        end
                    end
                    return trigger_list, #trigger_list;
                end
                """,
            )
            
            # merge trigger
            .if_("enable_short_term_trigger ~= 0")
                .pack_common_attr(
                    input_common_attrs=[
                        "pdn_item_trigger",
                        "user_profile_item_trigger",
                    ],
                    output_common_attr="trigger_list",
                    deduplicate=True,
                )
            .else_()
                .pack_common_attr(
                    input_common_attrs=[
                        "pdn_item_trigger",
                    ],
                    output_common_attr="trigger_list",
                    deduplicate=True,
                )
            .end_()

            # custom trigger: overrite trigger_list
            # .if_("custom_trigger ~= nil")
            #     .split_string(
            #         input_common_attr="custom_trigger",
            #         output_common_attr="custom_item_trigger",
            #         delimiters=",",
            #         parse_to_int=True,
            #         trim_spaces=True,
            #     )
            #     .copy_attr(
            #         attrs=[{
            #             "from_common": "custom_item_trigger",
            #             "to_common": "trigger_list"
            #         }] 
            #     )
            # .end_()
            # # custom i2i ann topk overrite i2i_ann_topk
            # .if_("custom_i2i_ann_topk ~= nil")
            #     .gen_common_attr_by_lua(
            #         attr_map={
            #             "i2i_ann_topk": "tonumber(custom_i2i_ann_topk)",
            #         }
            #     )
            # .end_()
            
        )

    def do_sub_list_retr(self, k=10):
        self._get_remote_trigger_emb()
        self.gen_common_attr_by_lua(
                attr_map={
                    "trigger_pid_num": "#(trigger_pid_list or {})"
                }
        )
        self.if_("trigger_pid_num <= 0").return_(2, "no trigger").end_if_()
        self.enrich_attr_by_lua(
            import_common_attr=["trigger_pid_list"],
            export_common_attr=["sub_trigger_pid_list_"+str(i) for i in range(k)],
            function_for_common="calc",
            lua_script=f"""
            function calc()
                local sub_trigger_pid_list = {{}};
                for i=1, {k} do
                    table.insert(sub_trigger_pid_list, {{}});
                end
                for i=1, #trigger_pid_list do
                    table.insert(sub_trigger_pid_list[i%{k}+1], trigger_pid_list[i]);
                end
                return table.unpack(sub_trigger_pid_list);
            end 
            """
        )
        self.enrich_attr_by_lua(
            import_common_attr=["trigger_pid_list", "trigger_emb_list"],
            export_common_attr=["sub_trigger_emb_list_"+str(i) for i in range(k)] ,
            function_for_common="calc",
            lua_script=f"""
            function calc()
                local sub_trigger_emb_list = {{}};
                for i=1, {k} do
                    table.insert(sub_trigger_emb_list, {{}});
                end
                for i=1, #trigger_pid_list do
                    for j=1, 512 do
                        table.insert(sub_trigger_emb_list[i%{k}+1], trigger_emb_list[(i-1)*512+j]);
                    end
                end
                return table.unpack(sub_trigger_emb_list);
            end 
            """
        )
        self.limit(0)
        for i in range(k):
            self.if_(f"sub_trigger_pid_list_{str(i)} ~= nil and #(sub_trigger_pid_list_{str(i)} or {{}}) > 0")
            self.delegate_retrieve(
                kess_service="{{i2i_ann_service_name}}",
                request_type="gpu_knn",
                send_common_attrs=[{"name": f"sub_trigger_pid_list_{str(i)}", "as": "photo_id_list"}, {"name": f"sub_trigger_emb_list_{str(i)}", "as": "photo_emb_list"},"i2i_ann_topk"],
                recv_item_attrs=[{"name": "ann_pid", "as": "trigger_pid"}],
                timeout_ms="{{i2i_ann_timeout}}",
                request_num=10000,
            )
            self.end_()
        return self

    def _get_remote_trigger_emb(self):
        return (
            self.retrieve_by_common_attr(
                attr="trigger_list",
                reason=1,
            )
            .copy_item_meta_info(
                save_item_id_to_attr="pid",
            )
            .get_remote_embedding(
                kess_service="grpc_mmu_e2e_i2i_PsCloud_online",
                slot = 0,
                shard_num=32,
                timeout_ms=10000,
                id_converter=dict(type_name="plainIdConverter"),
                output_attr_name="photo_emb",
                query_source_type="item_attr",
                query_source_item_attr="pid",
                save_to_common_attr=False,
                client_side_shard=True,
                is_raw_data=True,
                raw_data_type="float32",
            )
            .filter_by_attr(
                attr_name = "photo_emb",
                remove_if_attr_missing = True
            )
            .pack_item_attr(
                item_source={
                    "reco_results": True,
                },
                mappings=[
                    {
                        "aggregator": "concat",
                        "to_common_attr": "trigger_pid_list",
                    },
                    {
                        "aggregator": "concat",
                        "from_item_attr": "photo_emb",
                        "to_common_attr": "trigger_emb_list",
                    },
                ],
            )
        )


    def _retrieve(self, **kwargs):
        return (
            self
            .limit(0)
            .gen_common_attr_by_lua(
                attr_map={
                    "trigger_num": "#(trigger_list or {})"
                }
            )
            .if_("trigger_list ~= nil and trigger_num > 0")
            .shuffle_list_attr(common_attr="trigger_list")
            .truncate(size_limit="{{i2i_trigger_max_num}}", item_list_from_attr="trigger_list") \
            ._user_infer()
            .do_sub_list_retr(k=5)
            .count_reco_result(
                save_count_to="candidate_item_num",
            )
            .copy_item_meta_info(save_item_id_to_attr="pid")
            .filter_by_attr(
                attr_name="pid",
                remove_if="<=",
                compare_to=0,
                remove_if_attr_missing=True,
            )
            .filter_by_browse_set()
            .filter_by_common_attr(common_attr=["browsed_pids"])
            .deduplicate()
            .count_reco_result(save_count_to="filtered_item_num")
            .shuffle()
            .if_("enable_post_rank ~= 0")
               .post_rank() 
            .end_()
            .truncate(size_limit="{{limit_num}}")
            .count_reco_result(save_count_to="result_item_num")
            .perflog_attr_value(
                check_point=f"{self.name}.retr_stats",
                common_attrs=[
                    "trigger_num",
                    "i2i_trigger_max_num",
                    "user_profile_trigger_num",
                    "candidate_item_num",
                    "filtered_item_num",
                    "i2i_ann_topk",
                    "request_num",
                    "i2i_retr_num",
                    "limit_num",
                    "result_item_num",
                ],
            )
            .end_()
        )


flow = I2IRetrFlow(name="i2i_flow")._pre_process()._select_trigger()._retrieve()

service = LeafService(
    kess_name="grpc_slideI2ICenterServer-i2i-ia-ann",
    # common_attrs_from_request=["user", "custom_trigger", "custom_i2i_ann_topk"],
    common_attrs_from_request=["user"],
    # index_source=IndexSource.LOCAL_ATTR_INDEX
)
service.return_item_attrs(["trigger_pid"])

LeafService.CHECK_UNUSED_ATTR = False
service.add_leaf_flows(request_type="default", leaf_flows=[flow])

service.build(__file__.replace(".py", ".json"))
service.draw(dag_folder="./dag_folder")