#!/usr/bin/env python3
# coding=utf-8

import os, sys
import base64
import collections
import yaml

# from backbone import *
from dragonfly.common_leaf_dsl import LeafService, IndexSource, LeafFlow
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.pdn.pdn_api_mixin import PDNApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.uni_predict_v2.uni_predict_v2_api_mixin import UniPredictV2ApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin

kuiba_list_converter_config = {"converter": "list", "converter_args": {"reversed": False, "enable_filter": False}}
current_dir = os.path.dirname(__file__)
colossus_fields = [
    "photo_id",
    "author_id_v2",
    "timestamp",
    "label",
    "real_show_index",
    "tag",
    "duration",
    # "profile_feed_mode_stay_time",
    # "profile_stay_time",
    # "comment_stay_time",
    "play_time",
    "channel",
]

extra_colossus_fields = [
    "profile_feed_mode_stay_time",
    "profile_stay_time",
    "comment_stay_time",
]

EOL = '\n'

returned_user_seq_attrs = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010","1011","1012","1013","1014","colossus_time_s","1020","1021"] + ["1015", "1016", "1017", "1018", "1019"]
user_seq_slots = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010","1011","1012","1013","1014","1020","1021"] + ["1015", "1016", "1017", "1018", "1019"]

class GSUServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, CofeaApiMixin, EmbedCalcApiMixin, UniPredictApiMixin):
  def run(self):
    return (
      self.if_("use_user_seq == 1")
      .gpt_gsu(share=True, slot_whitelist={1001, 1002, 1004, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016}, extra_slots=False)
      .end_if_()
    )

  def gpt_gsu(self, share=False, filter_future_seconds=10 * 60, fix_colossus=False, slot_whitelist=None, extra_slots=False, include_extra_fields=False):
    colossus_features = colossus_fields + ["play_x_duration", "day_diff", "hour_in_day", "minute_diff", "position", "session_position", "hour_diff"]
    all_colossus_fields = colossus_fields

    self.if_("user_seq_size == nil")\
        .set_attr_value(
            no_overwrite=True,
            common_attrs=[
                {"name": "user_seq_size", "type": "int", "value": 512}
            ]
        ) \
        .end_if_()\
    
    if include_extra_fields:
      colossus_features = colossus_features + extra_colossus_fields
      all_colossus_fields = colossus_fields + extra_colossus_fields
    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "time_ms",
          "aggregator" : "min",
          "to_common_attr": "min_time_ms",
          "pack_if": "time_ms"
        }]
      ) \
      .enrich_attr_by_lua(
        import_common_attr=["_REQ_TIME_", "min_time_ms"],
        export_common_attr=["min_time_ms_zero", "request_gt_min", "request_gt600_min","request_diff"],
        function_for_common="calculate",
        lua_script="""
          function calculate()
            if min_time_ms == nil then
              return 1, 0, 0, 0
            end
            local diff = (_REQ_TIME_ - min_time_ms)/60000
            if diff >= 10 then
              return 0, 1, 1, diff
            end
            if diff >= 0 then
              return 0, 1, 0, diff
            end
            return 1, 0, 0, 0
          end
        """
      ) \
      .perflog_attr_value(check_point="recogpt.time_stat_diff", common_attrs=["min_time_ms_zero", "request_gt_min", "request_gt600_min"]) \
      .copy_user_meta_info(save_request_time_to_attr="origin_request_time") \
      .if_("request_gt_min > 0") \
        .log_debug_info(common_attrs=["_REQ_TIME_", "origin_request_time",  "min_time_ms", "request_gt_min","request_diff"], log_tag="before_truncate", respect_sample_logging=False) \
        .perflog_attr_value(check_point="recogpt.time_stat_diff", common_attrs=["request_diff"]) \
      .end_if_() \
      .reset_user_meta_info(timestamp_attr="min_time_ms", time_unit="ms") 

    self \
      .gsu_common_colossusv2_enricher(
          kconf="colossus.kconf_client.video_item",
          item_fields={field: f"colossus_{field}" for field in all_colossus_fields},
          filter_future_items=True,
          seconds_to_lookback=filter_future_seconds,
          limit="{{user_seq_size}}"
      ) \

    
    self \
      .copy_item_meta_info(save_item_seq_to_attr="item_seq") \
      .enrich_attr_by_lua(
        import_common_attr=["colossus_play_time", "colossus_duration", "_REQ_TIME_", "colossus_timestamp"],
        export_common_attr=["colossus_play_x_duration", "colossus_day_diff", "colossus_hour_in_day", "colossus_minute_diff", "colossus_position",  "colossus_session_position",  "colossus_hour_diff",  "last_time_gap", "session_num"],
        function_for_common ="calculate",
        lua_script="""
            function calculate()
                local play_x_duration = {}
                local day_diff = {}
                local hour_in_day = {}
                local minute_diff = {}
                local position = {}
                local sesssion_position = {}
                local hour_diff = {}
                
                local last_time_gap = 0
                local session_num = -1
                if colossus_play_time ~= nil and colossus_duration ~= nil and colossus_timestamp ~= nil then
                    local last_ts = 2<<31
                    for i = 1, #colossus_play_time do
                        index = #colossus_play_time - i + 1
                        local play_time = math.min(colossus_play_time[index], (1 << 24) - 1)
                        local duration = math.min(colossus_duration[index], (1 << 24) - 1)

                        play_x_duration[index] = (play_time << 24) + duration
                        day_diff[index] = (_REQ_TIME_ // 1000 - colossus_timestamp[index]) // (24 * 3600)
                        hour_diff[index] = ((_REQ_TIME_ // 1000 - colossus_timestamp[index]) // 3600) % 24
                        hour_in_day[index] = colossus_timestamp[index] // 3600 % 24
                        minute_diff[index] = math.max((_REQ_TIME_ // 1000 - colossus_timestamp[index]), 3600) // 60 % (24 * 60)
                        position[index] = i - 1
                        if (last_ts - colossus_timestamp[index]) > 10 then
                          session_num = session_num + 1
                        end
                        sesssion_position[index] = session_num
                        last_ts = colossus_timestamp[index]
                    end
                    last_time_gap = (_REQ_TIME_ // 1000 - colossus_timestamp[#colossus_timestamp]) / 60
                end
                return play_x_duration, day_diff, hour_in_day, minute_diff, position, sesssion_position, hour_diff, last_time_gap, session_num
            end
        """) \
      .log_debug_info(common_attrs=["_REQ_TIME_", "origin_request_time", "min_time_ms", "session_num"], log_tag="after_truncate", respect_sample_logging=False) \
      .perflog_attr_value(check_point="recogpt.new_time_stat", common_attrs=["last_time_gap", "session_num"]) \

    if share:
      self.if_("request_type == 'infer_request' or request_type == 'ntp_infer_request' or request_type == 'test_infer_request' or request_type == 'ntp_infer_request_7x64' or request_type == 'ntp_infer_request_3x8192'") \
        .extract_kuiba_parameter(
          config={
            f"extract_colossus_field_{field}": {
                "attrs": [{
                    "key_type": 26 if field == "photo_id" else 128 if field == "author_id_v2" else 1001 + field_idx,
                    "mio_slot_key_type": 1001 + field_idx,
                    # "key_type": 1001 + field_idx,
                    "attr": [f"colossus_{field}"],
                    **kuiba_list_converter_config,
                }],
            } for field_idx, field in enumerate(colossus_features) if slot_whitelist is None or 1001 + field_idx in slot_whitelist
          },
          is_common_attr=True,
          slots_output="user_seq_slots",
          parameters_output="user_seq_parameters"
        ) \
        .else_() \
        .extract_kuiba_parameter(
          config={
            f"extract_colossus_field_{field}": {
                "attrs": [{
                    "key_type": 26 if field == "photo_id" else 128 if field == "author_id_v2" else 1001 + field_idx,
                    "mio_slot_key_type": 1001 + field_idx,
                    # "key_type": 1001 + field_idx,
                    "attr": [f"colossus_{field}"],
                    **kuiba_list_converter_config,
                }],
            } for field_idx, field in enumerate(colossus_features) if slot_whitelist is None or 1001 + field_idx in slot_whitelist
          },
          # target_item={"item_seq": 0},
          is_common_attr=False,
          slot_as_attr_name=True
        ) \
        .end_()
    
      if extra_slots:
          self \
            .extract_kuiba_parameter(
              config={
                "extract_colossus_field_extra_photo_id": {"attrs": [{"key_type": 1020, "attr": ["colossus_photo_id"], **kuiba_list_converter_config}]},
                "extract_colossus_field_extra_author_id_v2": {"attrs": [{"key_type": 1021, "attr": ["colossus_author_id_v2"], **kuiba_list_converter_config}]},
              },
              target_item={"item_seq": 0},
              is_common_attr=False,
              slot_as_attr_name=True)
    else:
      self \
        .extract_kuiba_parameter(
          config={
            f"extract_colossus_field_{field}": {
                "attrs": [{
                    # "key_type": 26 if field == "photo_id" else 128 if field == "author_id_v2" else 1001 + field_idx,
                    # "mio_slot_key_type": 1001 + field_idx,
                    "key_type": 1001 + field_idx,
                    "attr": [f"colossus_{field}"],
                    **kuiba_list_converter_config,
                }],
            } for field_idx, field in enumerate(colossus_features) if slot_whitelist is None or 1001 + field_idx in slot_whitelist
          },
          target_item={"item_seq": 0},
          is_common_attr=False,
          slot_as_attr_name=True)
    self \
      .copy_attr(
        attrs=[{
          "from_common": "colossus_timestamp",
          "to_item": "colossus_time_s"
        }],
        target_item={"item_seq": 0},
      )\
      .copy_attr(
        attrs=[{
          "from_common": "colossus_timestamp",
          "to_common": "colossus_time_s"
        }]
      )

    for attr in colossus_features:
      self.log_debug_info(common_attrs=[f'colossus_{attr}'], for_debug_request_only=False)
      self.log_debug_info(common_attrs=[f'colossus_{attr}'], for_debug_request_only=False)
    return self

user_seq_flow = GSUServerFlow(name = "user_seq_flow").run()

# predict_for_gsu = GSUServerFlow(name = "predict_for_gsu").gpt_gsu(10000)
# predict_for_gsu_limit_10240_share_extra = GSUServerFlow(name = "predict_for_gsu_limit_10240_share_extra").gpt_gsu(10240, share=True, include_extra_fields=True, extra_slots=True)
# predict_for_gsu_limit_10240_share_fix = GSUServerFlow(name = "predict_for_gsu_limit_10240_share_fix").gpt_gsu(10240, share=True, slot_whitelist={1001, 1002, 1004, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016}, extra_slots=False)
# predict_for_gsu_limit_8192_share_fix_time = GSUServerFlow(name = "predict_for_gsu_limit_8192_share_fix_time").gpt_gsu(8192, share=True, slot_whitelist={1001, 1002, 1004, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014}, extra_slots=True)
# predict_for_gsu_limit_8192_share_pad = GSUServerFlow(name = "predict_for_gsu_limit_8192_share_pad").gpt_gsu(8192, share=True)
'''
service = LeafService(kess_name="grpc_sim3ComboFeatureServer",
                      item_attrs_from_request=["time_ms"],
                      common_attrs_from_request=["sim_video_item_photo_id", "sim_video_item_author_id",
                                                 "sim_video_item_duration", "sim_video_item_play_time",
                                                 "sim_video_item_tag", "sim_video_item_channel",
                                                 "sim_video_item_label", "sim_video_item_timestamp",
                                                 "sim_video_item_profile_stay_time", "sim_video_item_comment_stay_time",
                                                 "sim_video_item_profile_feed_mode_stay_time","sim_video_item_real_show_index"])
service.return_item_attrs(returned_user_seq_attrs)

service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False

service.add_leaf_flows(leaf_flows = [predict_for_gsu_share_fix], request_type = "predict_for_gsu", as_default=True)

if __name__ == '__main__':
  out_file = str(__file__).replace('py', 'json')
  service.build(output_file=os.path.join(current_dir, out_file))
'''