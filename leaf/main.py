#!/usr/bin/env python3
# coding=utf-8
import os
import sys
from dragonfly.common_leaf_dsl import LeafService
from prepare_flow import PrepareFlow
from send_log_flow import SendLogFlow
from direct_retrieval_flow import DirectRetrFlow
from main_process_flow import MainProcessFlow
from post_filter_flow import PostFilterFlow
from gen_result_flow import GenResultFlow

# 获取 ab 参数
prepare_flow = PrepareFlow(name="prepare")._default_flow()
# 直接召回结果
retr_flow = DirectRetrFlow(name="retr")._default_flow()
# 中间层处理逻辑，获取 photo_info
main_process_flow = MainProcessFlow(name="main_process")._default_flow()
# 发送日志，在filter 之前
send_log_flow = SendLogFlow(name="send_log")._default_flow()
# 后置过滤逻辑
filter_flow = PostFilterFlow(name="filter")._post_filter_flow()
# 生成最终结果
gen_result_flow = GenResultFlow(name="gen_result")._default_flow()

pxtrs = ["evtr", "lvtr", "ltr", "wtr", "wtd", "cmtr", "vtr", "svr", "cpr", "lsst", "epstr", "f1_score"]
return_common_attrs = ["user_id"]
return_item_attrs = ['retr_score', 'is_living'] + pxtrs

service = LeafService(kess_name="grpc_OnePieceLeafCkV1",
                      common_attrs_from_request=["user_info_attr", "source_photo_id", "page_size",
                                                 "tab_id", "is_tnu", "is_gamora_user", "is_nebula_user"])
service.return_common_attrs(return_common_attrs)
service.return_item_attrs(return_item_attrs)

service.CHECK_UNUSED_ATTR=False
service.IGNORE_NO_SOURCE_ATTR = ['score', 'is_nebula_user', 'is_gamora_user']
service.IGNORE_UNUSED_ATTR = ['user_info_ptr', 'request_id', "device_id", "request_time", "total_realshow_cnt", "duration_ms", 'session_score', 'retr_session_id', 'end2end_model_exp_name']

service.add_leaf_flows(leaf_flows=[prepare_flow,retr_flow, main_process_flow, send_log_flow, filter_flow, gen_result_flow], request_type="default", as_default=True)

# Build json config
current_folder = os.path.dirname(os.path.abspath(__file__))
service.build(output_file=os.path.join(current_folder, "dynamic_json_config.json"))
