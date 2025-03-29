from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin 
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin


class MFlow(LeafFlow, MioApiMixin, OfflineApiMixin, GsuApiMixin, PDNApiMixin):
    def _select_trigger(self, **kwargs):
        return (
            self
            # long term trigger
            .gsu_common_colossusv2_enricher(kconf="colossus.kconf_client.video_item",
                                    item_fields=dict(photo_id="photo_id_list",
                                                    author_id="author_id_list",
                                                    duration="duration_list",
                                                    play_time="play_time_list",
                                                    tag="tag_list",
                                                    label="label_list",
                                                    timestamp="timestamp_list"),
                                    limit=10000)
            .enrich_trigger_item(
                photo_id_from="photo_id_list",
                timestamp_from="timestamp_list",
                author_id_from="author_id_list",
                play_time_from="play_time_list",
                duration_from="duration_list",
                label_from="label_list",
                tag_from="tag_list",
                use_cluster_id=False,
                colossus_item_limit=10000,
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
                # missing_memory
                enable_missing_memory_trigger="{{enable_missing_memory_trigger}}",
                export_mm_cluster_photo_id="missing_cluster_photo_id",
                export_mm_author_photo_id="missing_author_photo_id",
                export_mm_author_cluster_photo_id="missing_author_cluster_photo_id",
                export_mm_missing_author_id="missing_author_id",
                export_mm_missing_author_cluster_aid="missing_author_cluster_aid",
                export_mm_recent_photo_cluster="recent_photo_cluster",
                export_mm_recent_author="recent_author",
                # export_mm_recent_author_cluster = "recent_author_cluster",
                mm_cluster_missing_day_num="{{cluster_missing_day_num}}",
                mm_cluster_valid_view_num="{{cluster_valid_view_num}}",
                mm_author_missing_day_num="{{author_missing_day_num}}",
                mm_author_valid_view_num="{{author_valid_view_num}}",
                mm_author_cluster_missing_day_num="{{author_cluster_missing_day_num}}",
                mm_author_cluster_valid_view_num="{{author_cluster_valid_view_num}}",
                mm_top_cluster_number="{{top_cluster_number}}",
                mm_photo_num_per_cluster="{{photo_num_per_cluster}}",
                mm_top_author_number="{{top_author_number}}",
                mm_photo_num_per_author="{{photo_num_per_author}}",
                mm_top_author_cluster_number="{{top_author_cluster_number}}",
                mm_photo_num_per_author_cluster="{{photo_num_per_author_cluster}}",            
            )
            # short term trigger
            .enrich_attr_by_lua( # 选择有效视频
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
            .if_("trigger_type == 'short'")
                .pack_common_attr(
                    input_common_attrs=[
                        "user_profile_item_trigger",
                    ],
                    output_common_attr="trigger_list",
                    deduplicate=True,
                )
            .else_() 
                .if_("trigger_type == 'long'")
                    .pack_common_attr(
                        input_common_attrs=[
                            "pdn_item_trigger",
                        ],
                        output_common_attr="trigger_list",
                        deduplicate=True,
                    )
                .else_()
                    .if_("trigger_type == 'short_and_long'")
                        .pack_common_attr(
                            input_common_attrs=[
                                "pdn_item_trigger",
                                "user_profile_item_trigger",
                                "swing_item_trigger",
                                "ltv_item_trigger",
                                "interact_item_trigger",
                            ],
                            output_common_attr="trigger_list",
                            deduplicate=True,
                        )
                    .end_()
                .end_()            
            .end_()
        )

fetch_trigger_flow = (
    MFlow("fetch_trigger_flow")
    .namespace_(ns="fetch_trigger_flow", nest=True)
    ._select_trigger() 
    .gen_common_attr_by_lua(
        attr_map={
            "trigger_num": "#(trigger_list or {})",
            "pdn_item_trigger_num": "#(pdn_item_trigger or {})",
        }  
    )
    .perflog_attr_value(
        check_point="{{return kconf_key .. '.fetch_trigger_flow'}}",
        common_attrs=[
            "trigger_type",
            "trigger_num",
            "pdn_item_trigger_num",
        ],
    )
    .namespace_()
)