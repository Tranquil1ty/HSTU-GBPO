#!/usr/bin/env python3
# coding=utf-8
from dragonfly.common_leaf_dsl import LeafFlow

# reco_one_rec_result_log
class SendLogFlow(LeafFlow):
    def __init__(self, name):
        LeafFlow.__init__(self, name)

    def _default_flow(self):
       return (
          self.if_("one_rec_enable_send_kafka_log == 1") 
          .export_attr_to_kafka(
            kafka_topic="reco_one_rec_result_log",
            common_attrs=["user_id", "device_id", "tab_id", "request_id", "current_time_ms", "request_time",
                          "is_gamora_user", "is_nebula_user", "direct_retr_kess_name"],
            item_attrs=["retr_score", "evtr", "ltr", "wtr", "ftr", "cmtr", "lvtr", 
                        "vtr", "svr", "ptr", "cmef", "qtr", "cpr", "lsst", "l2r_pxtr",
                        "onerec_mix_pxtr"],
            single_json=True
            )
          .end_if_()
       )
