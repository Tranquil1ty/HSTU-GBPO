#!/usr/bin/env python3
# coding=utf-8
from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.nr.nr_api_mixin import NrApiMixin

class PrepareFlow(LeafFlow, NrApiMixin):
  def __init__(self, name):
    LeafFlow.__init__(self, name)

  def _default_flow(self):
    self._pre_pare()
    self._get_abtest_params()
    self._process_ab_params()
    return self
    
  def _pre_pare(self):
    self.parse_protobuf_from_string(
      input_attr = "user_info_attr",
      output_attr = "user_info_ptr",
      class_name = "ks.reco.UserInfo",
    ) \
    .copy_user_meta_info(
      save_user_id_to_attr="user_id",
      save_device_id_to_attr="device_id",
      save_request_id_to_attr="request_id", # = std::to_string(llsid)
      save_request_time_to_attr="request_time",
      save_current_time_ms_to_attr="current_time_ms",
    ) \
    .enrich_with_protobuf(
      from_extra_var="user_info_ptr",
      attrs=[
          dict(name="city_id", path="request_location_new.city_id"),
          dict(name="black_author_list", path="black_author_list"),
          dict(name="risk_level_test", path="risk_level_test"),
          dict(name="hate_aid_list",path="user_profile_v1.hate_list.author_id",repeat_limit={"user_profile_v1.hate_list": 200}),
          dict(name="report_aid_list",path="user_profile_v1.report_list.author_id",repeat_limit={"user_profile_v1.report_list": 200}),
          dict(name="hate_time_list",path="user_profile_v1.hate_list.time_ms",repeat_limit={"user_profile_v1.hate_list": 200}),
          dict(name="video_playing_reason",path="user_profile_v1.video_playing_stat.reason",repeat_limit={"user_profile_v1.video_playing_stat": 50}),
          dict(name="hate_hetu_one_list",path="user_profile_v1.hate_list.hetu_tag_level_info.hetu_level_one",repeat_limit={"user_profile_v1.hate_list.hetu_tag_level_info.hetu_level_one": 1}),
          dict(name="hate_hetu_two_list",path="user_profile_v1.hate_list.hetu_tag_level_info.hetu_level_two",repeat_limit={"user_profile_v1.hate_list.hetu_tag_level_info.hetu_level_two": 1}),
          dict(name="hate_hetu_three_list",path="user_profile_v1.hate_list.hetu_tag_level_info.hetu_level_three",repeat_limit={"user_profile_v1.hate_list.hetu_tag_level_info.hetu_level_three": 1}),
      ]
    ) \
    .enrich_attr_by_lua(
      import_common_attr = ["hate_time_list", "hate_hetu_one_list", "hate_hetu_two_list", "hate_hetu_three_list", "current_time_ms"],
        export_common_attr=["one_day_hate_hetu_one_list", "one_day_hate_hetu_two_list", "one_day_hate_hetu_three_list"],
        function_for_common="calculate",
        lua_script='''
            function calculate()
              local hate_time_list = hate_time_list or {}
              local hate_hetu_one_list = hate_hetu_one_list or {}
              local hate_hetu_two_list = hate_hetu_two_list or {}
              local hate_hetu_three_list = hate_hetu_three_list or {}
              local one_day_hate_hetu_one_list = {}
              local one_day_hate_hetu_two_list = {}
              local one_day_hate_hetu_three_list = {}
              if #hate_time_list ~= #hate_hetu_one_list then 
                return one_day_hate_hetu_one_list, one_day_hate_hetu_two_list, one_day_hate_hetu_three_list
              else
                for i=1, #hate_time_list do 
                  if current_time_ms - hate_time_list[i] <= 86400000 then 
                    one_day_hate_hetu_one_list.insert(hate_hetu_one_list[i])
                    if hate_hetu_two_list[i] ~= nil then 
                      one_day_hate_hetu_two_list.insert(hate_hetu_two_list[i])
                    end 
                    if hate_hetu_three_list[i] ~= nil then 
                      one_day_hate_hetu_three_list.insert(hate_hetu_three_list[i])
                    end
                  end
                end
                return one_day_hate_hetu_one_list, one_day_hate_hetu_two_list, one_day_hate_hetu_three_list 
              end
            end
        ''') \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.thanos.slideRetrBlockAuthor",
        "export_common_attr": "slide_retr_block_author_list",
        "value_type": "set_int64"
      }]
    ) \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.thanos.commonFilterPhotosSet",
        "export_common_attr": "common_filter_photo_set",
        "value_type": "set_int64"
      }]
    )
  
  def _get_abtest_params(self):
    return (
      self.get_abtest_params(
        biz_name = "KUAISHOU_APPS",
        ab_params = [{
            "param_name": "audit_risk_immd_tag_filter_str_gamora",
            "param_type": "string",
            "default_value": "2147249,2147250,2147251,2147252,2147253,2147254,2147255,2185071"
        },
        {
            "param_name": "sexy_content_filter_str_gamora",
            "param_type": "string",
            "default_value": "2147489"
        },
        {
            "param_name": "impression_audit_second_level_black_tags",
            "param_type": "string",
            "default_value": "2000854,2000847"
        },
        {
            "param_name": "high_hot_audit_second_level_black_tags",
            "param_type": "string",
            "default_value": "2000001084"
        },
        {
            "param_name": "topk_audit_second_level_black_tags",
            "param_type": "string",
            "default_value": "2000001084"
        },
        {
            "param_name": "author_type_vv_thresh_type_string",
            "param_type": "string",
            "default_value": "5,6,7,9,700"
        },
        {
            "param_name": "audit_b_second_tag_expand_normal",
            "param_type": "int",
            "default_value": "2000860"
        },
        {
          "param_name": "enable_use_nebula_request_type_new",
          "param_type": "int",
          "default_value": 0
        },
        {
          "param_name": "one_rec_ab_model_group",
          "param_type": "string",
          "default_value": "default",
          "attr_name": "exp_name"
        },
        {
          "param_name": "one_rec_retr_timeout_ms",
          "param_type": "int",
          "default_value": 600,
          "attr_name": "direct_retr_timeout_ms"
        },
        {
          "param_name": "one_rec_retr_num",
          "param_type": "int",
          "default_value": 1024,
          "attr_name": "direct_retr_num"
        },
        {
          "param_name": "one_rec_retr_request_type",
          "param_type": "string",
          "default_value": "default",
          "attr_name": "direct_retr_request_type"
        },
        {
          "param_name": "one_rec_retr_kess_name",
          "param_type": "string",
          "default_value": "grpc_listll6_onerec_infer",
          "attr_name": "direct_retr_kess_name"
        }
        ],
        deduplicate=True,
        parallel_get=32
      )
      .get_abtest_params(
        biz_name = "KUAISHOU_APPS",
        ab_params = [
          ("fr_model_copy_kess_name", "grpc_hqg24q4ModelComboCopy"),
          ("reward_model_timeout", 1000),
          ("muw_frequency_control_item_gap_limit", 20),
          ("muw_enable_use_forumua_one_v2", 0),
          ("enable_frequency_control", 0),
          ("one_rec_enable_use_nebula_formula_one", 0),
          ("one_rec_enable_send_kafka_log", 0)
        ]
      )
    )
  
  def _process_ab_params(self):
    return (
      self.split_string_list(
          input_common_attr = "audit_risk_immd_tag_filter_str_gamora",
          output_common_attr = "audit_risk_immd_tag_list",
          delimiters=",",
          parse_to_int=True
      )
      .split_string_list(
          input_common_attr = "sexy_content_filter_str_gamora",
          output_common_attr = "sexy_content_filter_list",
          delimiters=",",
          parse_to_int=True
      )
      .split_string_list(
          input_common_attr = "impression_audit_second_level_black_tags",
          output_common_attr = "impression_audit_second_level_black_tag_list",
          delimiters=",",
          parse_to_int=True
      )
      .split_string_list(
          input_common_attr = "high_hot_audit_second_level_black_tags",
          output_common_attr = "high_hot_audit_second_level_black_tag_list",
          delimiters=",",
          parse_to_int=True
      )
      .split_string_list(
          input_common_attr = "topk_audit_second_level_black_tags",
          output_common_attr = "topk_audit_second_level_black_tag_list",
          delimiters=",",
          parse_to_int=True
      )
      .split_string_list(
          input_common_attr = "author_type_vv_thresh_type_string",
          output_common_attr = "author_type_vv_thresh_type_string_list",
          delimiters=",",
      )
    )
