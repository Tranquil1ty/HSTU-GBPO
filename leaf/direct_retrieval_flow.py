#!/usr/bin/env python3
# coding=utf-8
from dragonfly.common_leaf_dsl import LeafFlow

class DirectRetrFlow(LeafFlow):
  def __init__(self, name):
    LeafFlow.__init__(self, name)

  def _default_flow(self):
    self.get_ab_param()
    self._perf_req_info()
    self.do_retr_with_frequency_control()
    self._do_retr()
    self.count_reco_result(save_count_to="item_num") \
    .perflog(mode='interval', value='{{item_num}}', namespace='common.leaf', subtag='onerec', extra1='item_cnt_after_retr', extra2="{{exp_name}}")
    return self

  def _perf_req_info(self):
    self.perflog(mode='count',
                 namespace='common.leaf',
                 subtag='onerec',
                 extra1='direct_retr_req_cnt',
                 extra2="{{exp_name}}") \
  
  def get_ab_param(self):
    abtest_config = {
      "biz_name": "KUAISHOU_APPS",
      "ab_params": [
        ("one_piece_leaf_kconfig", "rinf.rlRunner.one_piece_leaf_kconfig"),
        ("enable_use_fr_model_copy_v2", False),
        ("fr_model_copy_kess_name", "grpc_hqg24q4ModelComboCopy"),
      ]
    }
    kconf_configs_var = {
      "direct_retr_kess_name": "grpc_listll6_onerec_infer",
      "direct_retr_request_type": "default",
      "direct_retr_num": 1024,
      "direct_retr_timeout_ms": 600,
      "exp_name": "default",
      "enable_frequency_control": 1,
      "reward_model_kess_name": "grpc_rljListScoreForGen",
      "reward_model_request_type": "predict_for_point",
      "reward_model_partition_size": 100,
      "reward_model_timeout": 1000,
      "muw_frequency_control_item_gap_limit": 20,
      "muw_enable_request_online_fr_model": 0,
      "muw_enable_use_forumua_one_v2": 0
    }
    kconf_params_config = [
      {"kconf_key": "{{one_piece_leaf_kconfig}}", "export_common_attr": k, "json_path": k, "default_value": v}
      for k, v in kconf_configs_var.items()
    ]
    self.get_abtest_params(**abtest_config)
    self.get_kconf_params(kconf_configs=kconf_params_config)

  def _do_retr(self):
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
        "bubble_sr", "search_page_photo_show", "search_page_photo_click"
    ]
    return self.if_("enable_frequency_control == 1 and frequency_enable == 0") \
    .return_(0) \
    .else_() \
    .delegate_retrieve(
        kess_service="{{direct_retr_kess_name}}",
        request_num="{{direct_retr_num}}",
        request_type="{{direct_retr_request_type}}",
        send_common_attrs=[{"name": "user_info_attr", "as" : "user"}, "is_nebula_user", "is_gamora_user"],
        recv_item_attrs=[{"name": "log_p", "as" : "retr_score"}], # retr score 为 log 形式，所以为负数
        timeout_ms='{{direct_retr_timeout_ms}}',
        reason=1024
    ) \
    .count_reco_result(save_count_to="item_num") \
    .perflog(mode='interval', value='{{item_num}}', namespace='common.leaf', subtag='onerec', extra1="item_cnt_retr", extra2="{{exp_name}}") \
    .end_if_()

  def do_retr_with_frequency_control(self):
    self.enrich_attr_by_lua(
      import_common_attr=["video_playing_reason", "muw_frequency_control_item_gap_limit"],
      export_common_attr=["frequency_enable"],
      function_for_common="calc",
      lua_script="""
        function calc()
          local reason = ''
          local start = -1
          local finish = -1
          for i = 1, #video_playing_reason do
            if i < muw_frequency_control_item_gap_limit then
              reason = video_playing_reason[i]
              start, finish = string.find(reason, '1024')
              if start ~= nil and start > 0 then
                return 0
              end
            end
          end
          return 1
        end
        """
    ) \
    .perflog(mode='interval',
      value="{{frequency_enable}}",
      namespace='common.leaf',
      subtag='onerec',
      extra1='frequency_enable',
      extra2="{{exp_name}}") \
