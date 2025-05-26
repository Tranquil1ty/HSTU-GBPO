#!/usr/bin/env python3
# coding=utf-8
from dragonfly.common_leaf_dsl import LeafFlow

class PostFilterFlow(LeafFlow):
  def __init__(self, name):
    LeafFlow.__init__(self, name)

  def _post_filter_flow(self):
    self.filter_by_browse_set() \
    .count_reco_result(save_count_to="item_num") \
    .perflog(mode='interval', value='{{item_num}}', namespace='common.leaf', subtag='onerec', extra1="item_num_by_browset", extra2="{{exp_name}}") \
    .filter_by_attr(
        attr_name="photo_status",
        remove_if=">",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="forbid_hot_page",
        remove_if=">",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .enrich_attr_by_lua(
      import_item_attr=['same_frame_sources'],
      export_item_attr=["is_same_frame_filter_photo"],
      function_for_item="calculate",
      lua_script=f'''
          function calculate()
              local is_same_frame_filter_photo = 0
              if same_frame_sources ~= nil and #same_frame_sources > 0 then 
                is_same_frame_filter_photo = 1 
              end
              return is_same_frame_filter_photo
          end
      ''') \
    .filter_by_attr(
        attr_name="is_same_frame_filter_photo",
        remove_if=">",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="long_term_high_level_photo",
        remove_if=">",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="inferior_first_review_score",
        remove_if=">",
        compare_to=0.89,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="is_jianguan_risk_photo",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="audit_hot_high_subdivision_level",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .enrich_attr_by_lua(
      import_common_attr=['city_id'],
      import_item_attr=['slide_punish_city'],
      export_item_attr=["is_city_punish_photo"],
      function_for_item="calculate",
      lua_script=f'''
          function calculate()
              local is_city_punish_photo = 0
              if (slide_punish_city ~= nil) then 
                for i = 1, #slide_punish_city do
                    if slide_punish_city[i] == city_id then
                        is_city_punish_photo = 1
                    end
                end
              end
              return is_city_punish_photo
          end
      ''') \
    .filter_by_attr(
        attr_name="is_city_punish_photo",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .enrich_attr_by_lua(
      import_common_attr=['user_id'],
      import_item_attr=['invisible_friends'],
      export_item_attr=["is_invisible_friend_photo"],
      function_for_item="calculate",
      lua_script=f'''
          function calculate()
              local is_invisible_friend_photo = 0
              if (invisible_friends ~= nil) then 
                for i = 1, #invisible_friends do
                    if invisible_friends[i] == user_id then
                        is_invisible_friend_photo = 1
                    end
                end
              end
              return is_invisible_friend_photo
          end
      ''') \
    .filter_by_attr(
        attr_name="is_invisible_friend_photo",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .enrich_attr_by_lua(
      import_common_attr=['risk_level_test'],
      import_item_attr=['photo_level'],
      export_item_attr=["should_photo_level_filter"],
      function_for_item="calculate",
      lua_script=f'''
          function calculate()
            local should_photo_level_filter = 0
            if risk_level_test >= 5 and photo_level < risk_level_test then 
              should_photo_level_filter = 1 
            end 
            return should_photo_level_filter
          end
      ''') \
    .filter_by_attr(
        attr_name="should_photo_level_filter",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="audit_hot_high_tag_level",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="level_hot_online",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="topk_audit_level",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="audit_b_second_tag",
        remove_if="==",
        compare_to='{{audit_b_second_tag_expand_normal}}',
        remove_if_attr_missing=True,
    ) \
    .filter_by_attr(
        attr_name="audit_b_second_tag",
        remove_if="==",
        compare_to=-999,
        remove_if_attr_missing=True,
    ) \
    .filter_by_browse_set(
      check_id_in_attr='dup_cluster_id',
      select_item = {
          "attr_name": "dup_cluster_id",
          "compare_to": 0,
          "select_if": ">",
          "select_if_attr_missing": False
      }
    ) \
    .filter_by_browse_set(
      check_id_in_attr='sim_remove_dup_id',
      select_item = {
          "attr_name": "sim_remove_dup_id",
          "compare_to": 0,
          "select_if": ">",
          "select_if_attr_missing": False
      }
    ) \
    .filter_by_browse_set(
      check_id_in_attr='pic_and_selfdup_id',
      select_item = {
          "attr_name": "pic_and_selfdup_id",
          "compare_to": 0,
          "select_if": ">",
          "select_if_attr_missing": False
      }
    ) \
    .filter_by_browse_set(
      check_id_in_attr='original_photo_id',
      select_item = {
          "attr_name": "original_photo_id",
          "compare_to": 0,
          "select_if": ">",
          "select_if_attr_missing": False
      }
    ) \
    .filter_by_common_attr(common_attr=["common_filter_photo_set"]) \
    .filter_by_common_attr(common_attr=["audit_risk_immd_tag_list"], on_item_attr='audit_risk_immd_tag') \
    .filter_by_common_attr(common_attr=["impression_audit_second_level_black_tag_list"], on_item_attr='audit_b_second_tag') \
    .filter_by_common_attr(common_attr=["high_hot_audit_second_level_black_tag_list"], on_item_attr='explore_operation_c_review_level') \
    .filter_by_common_attr(common_attr=["topk_audit_second_level_black_tag_list"], on_item_attr='topk_audit_tag') \
    .filter_by_common_attr(common_attr=["author_type_vv_thresh_type_string_list"], on_item_attr='catorgery_level_1') \
    .filter_by_common_attr(common_attr=["sexy_content_filter_list"], on_item_attr='audit_b_second_tag') \
    .filter_by_common_attr(common_attr=["hate_aid_list"], on_item_attr='author_id') \
    .filter_by_common_attr(common_attr=["hate_aid_list"], on_item_attr='author_big_id') \
    .filter_by_common_attr(common_attr=["report_aid_list"], on_item_attr='author_id') \
    .filter_by_common_attr(common_attr=["report_aid_list"], on_item_attr='author_big_id') \
    .filter_by_common_attr(common_attr=["black_author_list"], on_item_attr='author_id') \
    .filter_by_common_attr(common_attr=["black_author_list"], on_item_attr='author_big_id') \
    .filter_by_common_attr(common_attr=["slide_retr_block_author_list"], on_item_attr='author_id') \
    .filter_by_common_attr(common_attr=["one_day_hate_hetu_one_list"], on_item_attr='hetu_tag_level_one') \
    .filter_by_common_attr(common_attr=["one_day_hate_hetu_two_list"], on_item_attr='hetu_tag_level_two') \
    .filter_by_common_attr(common_attr=["one_day_hate_hetu_three_list"], on_item_attr='hetu_tag_level_three') \
    .if_("enable_markcodes_filter == 1") \
    .filter_by_attr(
        attr_name="is_need_filter_markcode_item",
        remove_if="==",
        compare_to=1,
        remove_if_attr_missing=False,
    ) \
    .end_if_() \
    .count_reco_result(save_count_to="item_num") \
    .perflog(mode='interval', value='{{item_num}}', namespace='common.leaf', subtag='onerec', extra1="item_cnt_after_filter", extra2="{{exp_name}}")
    return self