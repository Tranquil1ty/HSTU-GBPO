#!/usr/bin/env python3
# coding=utf-8
from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.slide.slide_api_mixin import SlideApiMixin

class MainProcessFlow(LeafFlow, SlideApiMixin):
  pxtrs = ["evtr", "lvtr", "ltr", "wtr", "wtd_v2", "cmtr", "vtr", "svr", "cpr", "lsst", "epstr"]
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
      "bubble_sr", "search_page_photo_show", "search_page_photo_click", "l2r_pxtr", "onerec_mix_pxtr", "onerec_mix_pxtr_gamora", "onerec_mix_pxtr_nebula", "onerec_mix_pxtr_all"
  ]
  def __init__(self, name):
    LeafFlow.__init__(self, name)
  
  def _default_flow(self):
    self.fetch_photo_info()
    return self

  def fetch_photo_info(self):
    # test
    self.get_item_attr_by_distributed_index(
      photo_store_kconf_key = "reco.distributedIndex.hotPhotoStoreConfig",
      use_dynamic_photo_store = True,
      attrs = [
        { "name": "author_id", "path": "author.id" },
        { "name": "author_big_id", "path": "author.big_id" },
        "duration_ms",
        "photo_status",
        "is_jianguan_risk_photo",
        "slide_punish_city",
        "dup_cluster_id",
        "sim_remove_dup_id",
        "pic_and_selfdup_id",
        "invisible_friends",
        "original_photo_id",
        "forbid_hot_page",
        "same_frame_sources",
        "inferior_first_review_score",
        "long_term_high_level_photo",
        "audit_hot_high_subdivision_level",
        "audit_risk_immd_tag",
        "audit_b_second_tag",
        "audit_hot_high_tag_level",
        "explore_operation_c_review_level",
        "topk_audit_level",
        "topk_audit_tag",
        { "name": "level_hot_online", "path": "content_safety_level_with_namespace.level_hot_online" },
        { "name": "catorgery_level_1", "path": "author_category.catorgery_level_1" },
        { "name": "photo_level", "path": "content_safety_level_with_namespace.hot_online_v2" },
        { "name": "hetu_tag_level_one_list", "path": "author.hetu_author_tag.hetu_level_one" },
        { "name": "hetu_tag_level_two_list", "path": "author.hetu_author_tag.hetu_level_two" },
        { "name": "hetu_tag_level_three_list", "path": "author.hetu_author_tag.hetu_level_three" },
        { "name": "nebula_realshow_cnt", "path": "nebula_stats.real_show_count" },
        { "name": "gamora_realshow_cnt", "path": "thanos_stats.real_show_count" },
        { "name": "is_living", "path": "live_photo_info.is_living" },
        { "name": "markcodes", "path": "sirius_distribution_info.mark_cod"}
      ]
    ) \
    .parse_hetu_tag(
      configs = [
        { "hetu_list_attr": "hetu_tag_level_one_list", "hetu_attr": "hetu_tag_level_one" },
        { "hetu_list_attr": "hetu_tag_level_two_list", "hetu_attr": "hetu_tag_level_two" },
        { "hetu_list_attr": "hetu_tag_level_three_list", "hetu_attr": "hetu_tag_level_three" },
      ],
      hetu_v2 = True,
      ignore_tags = [64]
    ) \
    .enrich_attr_by_lua(
      import_item_attr=['nebula_realshow_cnt', 'gamora_realshow_cnt'],
      export_item_attr=["total_realshow_cnt"],
      function_for_item="calculate",
      lua_script=f'''
          function calculate()
            local nebula_realshow_cnt = nebula_realshow_cnt or 0
            local gamora_realshow_cnt = gamora_realshow_cnt or 0
            return nebula_realshow_cnt + gamora_realshow_cnt
          end
      ''') \
    .copy_item_meta_info(
      save_item_key_to_attr="item_id",
    ) \
    .build_protobuf(  # 电商 直播链路在自己的链路里提前 build , 能拿到这些pxtr吗？
        inputs = [
            { "item_attr": "is_living", "path": "living" },
            { "item_attr": "item_id", "path": "ar_result.pid" },
        ],
        output_item_attr = "reco_photo_info_str",
        class_name = "ks::reco::RecoPhotoInfo",
        as_string = True
        ) \
    .if_("item_num > 0") \
      .set_attr_value(
        no_overwrite=True,
        common_attrs=[{
          "name": "retr_type",
          "type": "int",
          "value": 1
        }]
      ) \
      .set_attr_value(
        common_attrs=[{
          "name": "main_model_request_type",
          "type": "string",
          "value": "predict_for_gamora"
        }]
      ) \
      .if_("enable_use_nebula_request_type_new == 1 and is_nebula_user == 1") \
      .set_attr_value(
        common_attrs=[{
          "name": "main_model_request_type",
          "type": "string",
          "value": "predict_for_nebula"
        }]
      ) \
      .end_if_() \
      .delegate_enrich(
        name = "delegate_enrich_main_model_copy",
        kess_service="{{fr_model_copy_kess_name}}",
        request_type="{{main_model_request_type}}",
        timeout_ms="{{onerec_reward_model_timeout}}",
        send_common_attrs = [
            {"name": "user_info_attr", "as": "user_info_str"},
            "tab_id",
            "retr_type"
        ],
        send_item_attrs=[
            { "name": "reco_photo_info_str", "as": "reco_photo_info_str" },
            { "name": "is_living", "as": "living" },
        ],
        recv_item_attrs=self.main_model_pxtrs,
        partition_size=256,
        use_packed_item_attr = True
      ) \
      .enrich_attr_by_lua(
        import_item_attr=["wtd_v2"],
        export_item_attr=["wtd"],
        function_for_item="calc",
        lua_script="""
          function calc()
            local wtd = wtd_v2 or 0
            return wtd
          end
        """
      ) \
      .enrich_attr_by_lua(
        import_item_attr=["markcodes"],
        import_common_attr=["markcodes_filter_list"],
        export_item_attr=["is_need_filter_markcode_item"],
        function_for_item="filter",
        lua_script="""
          function filter()
            local markcodes = markcodes or {}
            local markcodes_filter_list = markcodes_filter_list or {}
            for _, markcode in ipairs(markcodes) do
              if table.contains(markcodes_filter_list, markcode) then
                return 1
              end
            end
            return 0
          end
        """
      ) \
      .perflog_attr_value(
        check_point="{{return 'onerec.need_filter_markcodes' .. exp_name}}",
        item_attrs= ["is_need_filter_markcode_item"],
        aggregator='avg'
      ) \
      .end_if_()
    return self
