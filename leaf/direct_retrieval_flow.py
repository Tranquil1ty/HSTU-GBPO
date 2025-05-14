#!/usr/bin/env python3
# coding=utf-8
from dragonfly.common_leaf_dsl import LeafFlow

class DirectRetrFlow(LeafFlow):
  def __init__(self, name):
    LeafFlow.__init__(self, name)

  def _default_flow(self):
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

  def _do_retr(self):
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
