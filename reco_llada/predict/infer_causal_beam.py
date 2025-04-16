import os

os.environ["DRAGON_MATX_USE_REMOTE_DSO"] = "true"

from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.common.common_api_mixin import CommonApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from dragonfly.decorators import for_loop

from infer_base import (
    photo_store_config, extra_reco_photo_info_attrs,
    load_model, extract_input_from_slot_config, kuiba_list_converter_config_limit50
)
from segment_fetcher_new import (
    GenSegmentConfigSaAN,
    USE_SELECT_SIGN_REPLACE_LUA
)

from sample_strategy import ChooseTokenStrategy, RemaskTokenStrategy
from hit_rate_perf import HitRatePerfFlow
from eval_flow import EvalFlow

TAB_NEBULA = 30000
kess_name = "grpc_llada_debug"

model_config = dict(
    model_config=load_model("./causal_large_config"),
    colossusdb_embd_service_name="reco_llada",
    colossusdb_embd_table_name="reco_ar_cuzhao_large_emb",
    embedding_shard_num=8,
    queue_prefix="reco_ar_cuzhao_large",
    embedding_dtype="scale_int8",
    common_slots_mapping=[
        # (704, 702), (704, 706), # wtd_v2
        (704, 702),
        (704, 706),
        (704, 707),
        (704, 708),
        (704, 709),
        (704, 710),
        (704, 904),  
        # mlga # 增加需要copy的slots
        (704, 1395),
        (1079, 2079),
        (184, 1184),
        (747, 1747),
        (752, 1752),
        (182, 1182),
        (757, 1757),
        (827, 887),
    ],
    item_slots_mapping=[(506, 507), (807, 867), (817, 877)],
    use_fp16=False,
    use_fused_op=True,
    context_per_device=6,
    executor_per_flavor=6,
    optimizers=[
        #'MatmulBiasaddReluFusion',
        # 'CascadeMatMulFusion',
    ],
    explicit_batchsizes=[6], #, 2, 8, 16, 32],
    explicit_max_enqueued_batches=1,
    use_tvm=False
)

extra_feature_list = list(sorted(set(model_config["model_config"].feature_list)))
print(extra_feature_list)

user_info_attrs = [
    dict(path="id", name="user_id"),
    "device_id",
    # dict(path="user_class.nr_user_type", name="nr_user_type"),
    dict(path="realtime_click_list", name="realtime_click_list"),
    dict(path="realtime_like_list", name="realtime_like_list"),
    dict(path="location.lat", name="user_lat"),
    dict(path="location.lon", name="user_lon"),
    dict(path="request_location_new.city_id", name="user_req_city_id"),
    # dict(path="basic_info.age_segment", name="user_age_segment"),
    dict(path="live_profile_v1.live_play_list.anchor_id", name="live_play_aid"),
    dict(path="live_profile_v1.live_play_list.live_play_time", name="live_play_time"),
    dict(path="live_profile_v1.live_play_list.client_timestamp", name="live_timestamp"),
    dict(path="extra_info.user_active_devices", name="user_active_devices"),
    dict(path="bind_user_id", name="bind_user_id"),
]

# 使用 kuiba 抽取特征
kuiba_parameter_common_config = {
  "cpr_did": {"attrs": [{"key_type": 738, "attr": ["device_id"], "converter": "id"}]},
  "mmu_did": {"attrs": [{"key_type": 839, "attr": ["device_id"], "converter": "id"}]},
  "cascade_uid": {"attrs": [{"key_type": 851, "attr": ["user_id"], "converter": "id"}]},
  "cascade_did": {"attrs": [{"key_type": 852, "attr": ["device_id"], "converter": "id"}]},
  "user_geohash_id": {"attrs": [{"key_type": 1530, "attr": ["user_id"], "converter": "id"}]},
  "device_geohash_id": {"attrs": [{"key_type": 1531, "attr": ["device_id"], "converter": "id"}]},
  "user_geohash_2": {"attrs": [{"key_type": 1532, "attr": ["user_lat", "user_lon"], "converter": "geohash", "converter_args": "2"}]},
  "user_geohash_3": {"attrs": [{"key_type": 1533, "attr": ["user_lat", "user_lon"], "converter": "geohash", "converter_args": "3"}]},
  "user_geohash_4": {"attrs": [{"key_type": 1534, "attr": ["user_lat", "user_lon"], "converter": "geohash", "converter_args": "4"}]},
  "user_geohash_5": {"attrs": [{"key_type": 1535, "attr": ["user_lat", "user_lon"], "converter": "geohash", "converter_args": "5"}]},
  'user_req_city_id': {'attrs': [{'key_type': 1603, "mio_slot_key_type": 1603, 'attr': ['user_req_city_id'], "converter": "id"}]},
  "fuse_did": {"attrs": [{"key_type": 1391, "attr": ["device_id"], "converter": "id"}]},
  'user_live_aid': {'attrs': [{'key_type': 128, "mio_slot_key_type": 1611, 'attr': ['live_aid_list'], **kuiba_list_converter_config_limit50}]},
  'user_live_play': {'attrs': [{"key_type": 1615, 'attr': ['live_play_list'], **kuiba_list_converter_config_limit50}]},
  'user_live_timestamps': {'attrs': [{"key_type": 1616, 'attr': ['live_timestamps_list'], **kuiba_list_converter_config_limit50}]},
  "search_uid": {"attrs": [{"key_type": 101, "attr": ["user_id"], "converter": "id"}]},
  "search_did": {"attrs": [{"key_type": 102, "attr": ["device_id"], "converter": "id"}]},
  "adp_uid": {"attrs": [{"key_type": 566, "attr": ["user_id"], "converter": "id"}]},
  "adp_did": {"attrs": [{"key_type": 568, "attr": ["device_id"], "converter": "id"}]},
}

class PredictServerFlow(LeafFlow, CommonApiMixin, KuibaApiMixin, MioApiMixin, UniPredictApiMixin, EmbeddingApiMixin):


    def prepare_infer_params(self):
        # https://kconf.corp.kuaishou.com/#/reco/model2/reco_llada_params
        self.get_kconf_params(
            kconf_configs=[
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "adp_effective_view_fix",
                    "json_path": "adp_effective_view_fix",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "click",
                    "json_path": "click",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "like",
                    "json_path": "like",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "follow",
                    "json_path": "follow",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "forward",
                    "json_path": "forward",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "comment",
                    "json_path": "comment",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "long_view",
                    "json_path": "long_view",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "short_view",
                    "json_path": "short_view",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "play_complete",
                    "json_path": "play_complete",
                    "default_value": [0.0],
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "p_topk",
                    "json_path": "p_topk",
                    "default_value": 10,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "p_topp",
                    "json_path": "p_topp",
                    "default_value": 1.0,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "p_temp",
                    "json_path": "p_temp",
                    "default_value": 1.0,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "random_remask_flag",
                    "json_path": "random_remask_flag",
                    "default_value": 0,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "i2i_ann_topk",
                    "json_path": "i2i_ann_topk",
                    "default_value": 10,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "infer_step",
                    "json_path": "infer_step",
                    "default_value": 8,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "fake_item_num",
                    "json_path": "fake_item_num",
                    "default_value": 6,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "rag_service_kess_name",
                    "json_path": "rag_service_kess_name",
                    "default_value": "grpc_item_rag_user_longterm",
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "rag_service_timeout_ms",
                    "json_path": "rag_service_timeout_ms",
                    "default_value": 300,
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "reward_service_kess_name",
                    "json_path": "reward_service_kess_name",
                    "default_value": "grpc_hqg24q4ModelComboFinal",
                },
                {
                    "kconf_key": "reco.model2.reco_llada_params",
                    "export_common_attr": "use_ann",
                    "json_path": "use_ann",
                    "default_value": 0,
                }
            ]
        )

    def extract_from_feature_list(self, feature_list, is_common=False, **kwargs):
        need_source_photo_info = False

        return self \
          .extract_with_ks_sign_feature(
            feature_list=feature_list,
            tab_id_attr="tab_id",
            click_pids_attr="realtime_click_list",
            like_pids_attr="realtime_like_list",
            user_info_attr="user_info",
            **({} if is_common else dict(
              **(dict(source_photo_info_attr="source_photo_info") if need_source_photo_info else {}),
              photo_info_attr="photo_info",
              context_info_attr="context_info",
              retrieve_info_attr="retrieve_info",
              is_living_attr="living",
              pctr_attr="context_pctr",
              reason_attr="reason",
              page_attr="page")),
            **kwargs)

    def extract_live_fea(self):
        return self.enrich_attr_by_lua(
            import_common_attr=[
                "live_play_aid",
                "live_play_time",
                "live_timestamp",
                "_REQ_TIME_",
            ],
            export_common_attr=[
                "live_aid_list",
                "live_play_list",
                "live_timestamps_list",
            ],
            function_for_common="calculate",
            lua_script="""
                function calculate()
                local live_aid_list = {}
                local live_play_list = {}
                local live_timestamps_list = {}

                local aid_list = live_play_aid or {}
                local play_list = live_play_time or {}
                local time_ms_list = live_timestamp or {}

                for idx, value in ipairs(aid_list) do
                    aid = aid_list[idx]
                    play      = math.min(math.floor(play_list[idx] / 1000), 1000)
                    ts_m      = math.max(1.0, (_REQ_TIME_ - time_ms_list[idx])/ 60000.0)

                    table.insert(live_aid_list,aid)
                    table.insert(live_play_list, play)
                    table.insert(live_timestamps_list, math.floor(math.min(math.log(ts_m), 15.0)))
                end
                return live_aid_list, live_play_list, live_timestamps_list
                end
            """,
        )

    def reallocate_slots(self, **kwargs):
        configs = kwargs.pop("configs")
        slots_input_attr = kwargs.pop("slots_input")
        signs_input_attr = kwargs.pop("parameters_input")
        is_common_attr = kwargs.get("is_common_attr", False)
        enable_filter = kwargs.pop("enable_filter", True)

        # lixinhua03
        slot_as_attr_name = kwargs.pop("slot_as_attr_name", False)
        slot_as_attr_name_prefix = kwargs.pop("slot_as_attr_name_prefix", "")
        input_slots = list(set(c["input_slot"] for c in configs))

        if slot_as_attr_name == True:
            kuiba_parameter_config = dict()
            for c in configs:
                input_slot = c["input_slot"]
                output_slot = c["output_slot"]
                share_slot = c.get("share_slot", output_slot)
                kuiba_parameter_config[
                    f"reallocate_{output_slot}_from_{input_slot}"
                ] = {
                    "attrs": [
                        {
                            "key_type": share_slot,
                            "mio_slot_key_type": output_slot,
                            # 'attr': [f"signs_{input_slot}"],
                            "attr": [slot_as_attr_name_prefix + str(input_slot)],
                            "converter": "list",
                            "converter_args": {
                                "reversed": False,
                                "enable_filter": enable_filter,
                            },
                        }
                    ],
                }
            return self.extract_kuiba_parameter(config=kuiba_parameter_config, **kwargs)

        lua_script = f"""
            function calculate()
            local slots_input = _G["{slots_input_attr}"]
            local signs_input = _G["{signs_input_attr}"]
            local mapping_slots_to_signs = {{}}
            if not slots_input or not signs_input or #slots_input ~= #signs_input then
                return {", ".join(["nil"] * len(input_slots))}
            end

            for i = 1,#slots_input do
                local slot = slots_input[i]
                if {" or ".join(f"slot == {slot}" for slot in input_slots)} then
                if mapping_slots_to_signs[slot] == nil then
                    mapping_slots_to_signs[slot] = {{}}
                end
                local signs = mapping_slots_to_signs[slot]
                signs[#signs + 1] = signs_input[i]
                end
            end
            return {", ".join(f"mapping_slots_to_signs[{slot}]" for slot in input_slots)}
            end
        """
        if USE_SELECT_SIGN_REPLACE_LUA == False:
            if is_common_attr:
                lua_config = dict(
                    import_common_attr=[slots_input_attr, signs_input_attr],
                    export_common_attr=[
                        f"signs_{input_slot}" for input_slot in input_slots
                    ],
                    function_for_common="calculate",
                    lua_script=lua_script,
                )
            else:
                lua_config = dict(
                    import_item_attr=[slots_input_attr, signs_input_attr],
                    export_item_attr=[
                        f"signs_{input_slot}" for input_slot in input_slots
                    ],
                    function_for_item="calculate",
                    lua_script=lua_script,
                )
            self.enrich_attr_by_lua(**lua_config)
        else:
            self.select_sign(
                input_slot_attr=slots_input_attr,
                input_sign_attr=signs_input_attr,
                select_slots=[slot for slot in input_slots],
                output_sign_attrs=[f"signs_{slot}" for slot in input_slots],
                # reserve_size = 10,
                is_common_attr=is_common_attr,
            )

        kuiba_parameter_config = dict()
        for c in configs:
            input_slot = c["input_slot"]
            output_slot = c["output_slot"]
            share_slot = c.get("share_slot", output_slot)
            kuiba_parameter_config[f"reallocate_{output_slot}_from_{input_slot}"] = {
                "attrs": [
                    {
                        "key_type": share_slot,
                        "mio_slot_key_type": output_slot,
                        "attr": [f"signs_{input_slot}"],
                        "converter": "list",
                        "converter_args": {
                            "reversed": False,
                            "enable_filter": enable_filter,
                        },
                    }
                ],
            }

        return self.extract_kuiba_parameter(config=kuiba_parameter_config, **kwargs)

    def copy_features(self, **kwargs):
        common_slots_mapping = kwargs.pop("common_slots_mapping", [])
        item_slots_mapping = kwargs.pop("item_slots_mapping", [])
        list_enable_filter = kwargs.pop("list_enable_filter", True)

        # lixinhua03
        slot_as_attr_name = kwargs.pop("slot_as_attr_name", False)
        slot_as_attr_name_prefix = kwargs.pop("slot_as_attr_name_prefix", "")

        if common_slots_mapping:
            common_slots_input = kwargs.pop("common_slots_input")
            common_parameters_input = kwargs.pop("common_parameters_input")
            common_slots_output = kwargs.pop("common_slots_output")
            common_parameters_output = kwargs.pop("common_parameters_output")
            self.reallocate_slots(
                configs=[
                    dict(input_slot=input_slot, output_slot=output_slot)
                    for input_slot, output_slot in common_slots_mapping
                ],
                is_common_attr=True,
                slots_input=common_slots_input,
                parameters_input=common_parameters_input,
                slots_output=common_slots_output,
                parameters_output=common_parameters_output,
                enable_filter=list_enable_filter,
                slot_as_attr_name=slot_as_attr_name,
                slot_as_attr_name_prefix=slot_as_attr_name_prefix,
            )
        if item_slots_mapping:
            item_slots_input = kwargs.pop("item_slots_input")
            item_parameters_input = kwargs.pop("item_parameters_input")
            item_slots_output = kwargs.pop("item_slots_output")
            item_parameters_output = kwargs.pop("item_parameters_output")
            self.reallocate_slots(
                configs=[
                    dict(input_slot=input_slot, output_slot=output_slot)
                    for input_slot, output_slot in item_slots_mapping
                ],
                is_common_attr=False,
                slots_input=item_slots_input,
                parameters_input=item_parameters_input,
                slots_output=item_slots_output,
                parameters_output=item_parameters_output,
                enable_filter=list_enable_filter,
                slot_as_attr_name=slot_as_attr_name,
                slot_as_attr_name_prefix=slot_as_attr_name_prefix,
            )
        return self

    def get_uid_did(self):
        return self.enrich_attr_by_lua(
            import_common_attr=["_USER_ID_", "_DEVICE_ID_"],
            function_for_common="calculate",
            export_common_attr=["uid_did", "hash_did"],
            lua_script="""
                function calculate()
                if _USER_ID_ == 0 then
                    return util.CityHash64(_DEVICE_ID_), util.CityHash64(_DEVICE_ID_)
                end
                return _USER_ID_, util.CityHash64(_DEVICE_ID_)
                end
            """,
        ).log_debug_info(
            common_attrs=["_DEVICE_ID_", "_USER_ID_", "uid_did"],
            for_debug_request_only=True,
        )

    def set_tab_id(self, tab_id):
        return self.set_default_value(
            no_overwrite=True,
            common_attrs=[
                dict(name="tab_id", type="int", value=tab_id),
            ],
        ).perflog_attr_value(check_point="fullrank.tab_id", common_attrs=["tab_id"])

    def prepare_fake_item(self):
        self.enrich_attr_by_lua(
            import_common_attr=["gen_item_num", "fake_item_num"],
            function_for_common="calculate",
            export_common_attr=["fake_item_num"],
            lua_script="""
                function calculate()
                    if gen_item_num ~= nil then
                        return gen_item_num
                    else
                        return fake_item_num
                    end
                end
            """
        )
        self.fake_retrieve(num="{{fake_item_num}}", reason=777)

    def prepare_user_info(self):
        self.if_("user_info_str ~= nil")
        self.parse_protobuf_from_string(
            is_common_attr=True,
            input_attr="user_info_str",
            output_attr="user_info",
            class_name="ks::reco::UserInfo",
        )
        self.else_if_("user ~= nil")
        self.parse_protobuf_from_string(
            is_common_attr=True,
            input_attr="user", # eval_request
            output_attr="user_info",
            class_name="ks::reco::UserInfo",
        )
        self.else_()
        self.return_(1, "user_info is nil")
        self.end_if_()
        self.hot_fix_user_info(user_info_attr="user_info")
        # self.enrich_with_protobuf(
        #     from_extra_var="user_info",
        #     is_common_attr=True,
        #     attrs=[
        #         dict(path="user_profile_v1", name="user_profile"),
        #     ],
        # )
        self.enrich_with_protobuf(
            from_extra_var="user_info",
            is_common_attr=True,
            name="extract_user_info",
            attrs=user_info_attrs,
        )
        # self.enrich_attr_by_lua(
        #     import_common_attr=["user_active_devices"],
        #     function_for_common="calculate",
        #     export_common_attr=[
        #         "user_active_device1",
        #         "user_active_device2",
        #         "user_active_device3",
        #     ],
        #     lua_script="""
        #       function calculate()
        #         user_active_devices = user_active_devices or {}
        #         if #user_active_devices >= 3 then
        #           return util.CityHash64(user_active_devices[1]), util.CityHash64(user_active_devices[2]), util.CityHash64(user_active_devices[3])
        #         elseif #user_active_devices == 2 then
        #           return util.CityHash64(user_active_devices[1]), util.CityHash64(user_active_devices[2]), nil
        #         elseif #user_active_devices == 1 then
        #           return util.CityHash64(user_active_devices[1]), nil, nil
        #         else
        #           return nil, nil, nil
        #         end
        #       end
        #     """,
        # )
        self.extract_live_fea()
        self.log_debug_info(
            common_attrs=[
                "user_active_devices",
                "bind_user_id",
                "user_id",
                "device_id",
            ],
            for_debug_request_only=True,
        )

    def enrich_infer_labels(self):
        self.enrich_attr_by_lua(
            import_common_attr=[
                "adp_effective_view_fix", "click", "like", 
                "follow", "forward", "comment", "long_view", 
                "short_view", "play_complete",
                "p_temp"],
            function_for_item="calculate",
            export_item_attr=['adp_effective_view_fix', "click", "like", 
                                "follow","forward","comment","long_view",
                                "short_view","play_complete", "p_temp"],
            lua_script="""
                function calculate()
                    local item_adp_effective_view_fix = adp_effective_view_fix or {1.0}
                    local item_click = click or {0.0}
                    local item_like = like or {0.0}
                    local item_follow = follow or {0.0}
                    local item_forward = forward or {0.0}
                    local item_comment = comment or {0.0}
                    local item_long_view = long_view or {0.0}
                    local item_short_view = short_view or {0.0}
                    local item_play_complete = play_complete or {0.0}
                    local item_p_temp = {p_temp}
                    return item_adp_effective_view_fix, item_click, item_like, item_follow, item_forward, item_comment, item_long_view, item_short_view, item_play_complete, item_p_temp
                end
            """,
        )

    def enrich_decode_params(self):
        self.enrich_attr_by_lua(
            import_common_attr=["random_remask_flag", "p_topk", "p_topp", "p_temp"],
            export_common_attr=["random_remask_flag", "p_topk", "p_topp", "p_temp"],
            function_for_common="calculate",
            lua_script="""
                function calculate()
                    local random_remask_flag = random_remask_flag or 0
                    local p_topk = p_topk or 10
                    local p_topp = p_topp or 1.0
                    local p_temp = p_temp or 1.0
                    return random_remask_flag, p_topk, p_topp, p_temp
                end
            """,
        )
        self.log_debug_info(
            log_tag="infer_params",
            common_attrs=["random_remask_flag", "p_topk", "p_topp", "p_temp"],
            for_debug_request_only=True,
        )

    def extract_feature(self):
        self.enrich_infer_labels()
        self.enrich_decode_params()
        self.extract_kuiba_parameter(
            config=kuiba_parameter_common_config,
            is_common_attr=True,
            slots_output="kuiba_common_slots",
            parameters_output="kuiba_common_signs",
        )
        self.extract_from_feature_list(
            feature_list=extra_feature_list,
            is_common=True,
            common_slots_output="common_slots",
            common_parameters_output="common_signs",
        )
        self.log_debug_info(
            common_attrs=["common_slots", "common_signs", "kuiba_common_slots", "kuiba_common_signs", "user_info"],
            for_debug_request_only=True,
            # respect_sample_logging=False
        )
        self.copy_features(
            common_slots_input="common_slots",
            common_parameters_input="common_signs",
            common_slots_mapping=model_config["common_slots_mapping"],
            common_slots_output="copy_common_slots",
            common_parameters_output="copy_common_signs",
            list_enable_filter=False,
        )
        return self

    def get_photo_info(
        self,
        use_new_index=False,
        distributed_index="reco.distributedIndex.frHotPhotoStoreConfig",
        hot_fix_user_info=False,
        extra_reco_photo_info_attrs=[],
    ):
        if use_new_index:
            self.get_item_attr_by_distributed_new_photo_info_index(
                photo_store_kconf_key=distributed_index,
                save_item_info_to_attr="photo_info",
            )
        else:
            self.get_item_attr_by_distributed_index(
                photo_store_kconf_key=distributed_index,
                save_item_info_to_attr="photo_info",
            )

        self.parse_protobuf_from_string(
            is_common_attr=False,
            input_attr="reco_photo_info_str",
            output_attr="reco_photo_info",
            class_name="ks::reco::RecoPhotoInfo",
        )

        self.filter_by_attr(attr_name="photo_info", remove_if_attr_missing=True)

        self.enrich_with_protobuf(
            from_extra_var="reco_photo_info",
            is_common_attr=False,
            attrs=[
                dict(path="reason", name="reason"),
                dict(path="context_info", name="context_info"),
                dict(path="ar_result", name="retrieve_info"),
            ]
            + extra_reco_photo_info_attrs,
        )

        return self

    def item_rag(self):
        self.copy_attr(
            attrs=[{
                "from_common": "_REQ_TIME_",
                "to_item": "time_ms"
            }]
        )
        self.cast_attr_type(
            attr_type_cast_configs=[
                {
                "to_type": "int",
                "from_item_attr": "semantic_id_v2",
                "to_item_attr": "semantic_id_v2_int"
                },
            ]
        )    
        self.delegate_enrich(
            name="onerec_rag",
            kess_service="{{rag_service_kess_name}}",
            timeout_ms="{{rag_service_timeout_ms}}",
            send_item_attrs = [
                {"name": "semantic_id_v2_int", "as": "semantic_id_v2"},
                "time_ms"
            ],
            recv_item_attrs=["item_rag_slots", "item_rag_parameters", "ann_pids"],
            recv_common_attrs=["user_seq_slots", "user_seq_parameters", "colossus_time_s"],
            request_type="infer_request",
        )
        self.if_("request_type == 'debug_request'")
        self.pack_item_attr(
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "from_item_attr": "item_rag_slots",
                    "to_common_attr": "item_rag_slots",
                },
                {
                    "from_item_attr": "item_rag_parameters",
                    "to_common_attr": "item_rag_parameters",
                }
            ]
        )
        self.end_if_()
        self.if_("request_type == 'eval_request' or request_type == 'debug_request'")
        self.pack_item_attr(
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "from_item_attr": "ann_pids",
                    "to_common_attr": "all_ann_pids",
                }
            ]
        )
        self.copy_item_meta_info(item_list_from_attr="all_ann_pids", save_item_key_to_attr='photo_id')
        self.fetch_remote_embedding(
            item_list_from_attr="all_ann_pids",
            protocol=1,
            colossusdb_embd_model_name="rlj-24q2-norm-exp",
            colossusdb_embd_table_name="parallel_semantic_id",
            id_converter={"type_name":"plainIdConverter"},
            input_attr_name="photo_id",
            output_attr_name="ann_semantic_id_v2",
            query_source_type="item_attr",
            is_raw_data=True,
            raw_data_type="uint16",
            timeout_ms=10,
            size=16
        )
        self.pack_item_attr(
            item_source = {
                "reco_results": False,
                "common_attr": ["all_ann_pids"],
            },
            mappings = [{
                "from_item_attr": "ann_semantic_id_v2",
                "to_common_attr": "all_ann_semantic_id_v2",
            }]
        )
        self.enrich_attr_by_lua(
            import_common_attr = ['all_ann_semantic_id_v2'],
            import_item_attr = ['semantic_id_v2'],
            function_for_item="calculate",
            export_item_attr=['h_dist'],
            lua_script="""
                function calculate(seq, item_key, reason, score)
                    local hamming_dist = 0.0
                    for i = seq*32*16+1,(seq+1)*32*16 do
                        idx = (i % 16 - 1) % 16 + 1    -- 1~0 to 1~16
                        if all_ann_semantic_id_v2[i] == semantic_id_v2[idx] then
                            hamming_dist = hamming_dist + 1.0
                        end
                    end
                    return hamming_dist / 32.0
                end
            """
        )
        self.log_debug_info( 
            item_attrs = ["h_dist", "semantic_id_v2", "ann_pids"], 
            common_attrs = ["all_ann_semantic_id_v2"], 
            for_debug_request_only=True, 
            respect_sample_logging=False
        )
        self.perflog_attr_value(
            check_point="{{return 'reco_llada.h_dist_step' .. current_step}}",
            item_attrs=["h_dist"],
        )
        self.end_if_()
        return self

    def infer(self):
        colossusdb_embd_service_name = model_config["colossusdb_embd_service_name"]
        colossusdb_embd_table_name = model_config["colossusdb_embd_table_name"]
        slots_inputs = ["item_rag_slots"]
        parameters_inputs = ["item_rag_parameters"]
        common_slots_inputs = [
            "common_slots",
            "copy_common_slots",
            "kuiba_common_slots",
            "user_seq_slots"
        ]
        common_parameters_inputs = [
            "common_signs",
            "copy_common_signs",
            "kuiba_common_signs",
            "user_seq_parameters"
        ]
        embedding_dtype = model_config["embedding_dtype"]
        model = model_config["model_config"]
        use_fp16 = model_config.get("use_fp16", False)
        explicit_batchsizes = model_config.get("explicit_batchsizes", [1])
        explicit_max_enqueued_batches = model_config.get(
            "explicit_max_enqueued_batches", 1
        )
        embedding_shard_num = model_config.get("embedding_shard_num", 8)
        optimizers = model_config.get("optimizers", [])
        use_tvm = model_config.get("use_tvm", False)
        context_per_device = model_config.get("context_per_device", 12)
        output_prefix = model_config.get("output_prefix", "")

        fetch_general_embedding = dict(
            colossusdb_embd_service_name=colossusdb_embd_service_name,
            colossusdb_embd_table_name=colossusdb_embd_table_name,
            protocol=0,
            timeout_ms=30,
            slots_inputs=slots_inputs,
            parameters_inputs=parameters_inputs,
            common_slots_inputs=common_slots_inputs,
            common_parameters_inputs=common_parameters_inputs,
            slots_config=[
                dict(dtype=embedding_dtype, **sc) for sc in model.slots_config
            ],
            max_signs_per_request=500,
            fetcher_type="ColossusdbEmbeddingServerFetcher",
        )
        fused_input = [extract_input_from_slot_config(c) for c in model.slots_config]
        fused_input.append(dict(attr_name="semantic_id_v2", 
                                tensor_name="semantic_id_v2", common=False, dim=16))
        fused_input.append(dict(attr_name="adp_effective_view_fix",
                                tensor_name="adp_effective_view_fix", common=False, dim=1))

        fused_input.append(dict(attr_name="click", tensor_name="click", common=False, dim=1))
        fused_input.append(dict(attr_name="like", tensor_name="like", common=False, dim=1))
        fused_input.append(dict(attr_name="follow", tensor_name="follow", common=False, dim=1))
        fused_input.append(dict(attr_name="forward", tensor_name="forward", common=False, dim=1))
        fused_input.append(dict(attr_name="comment", tensor_name="comment", common=False, dim=1))
        fused_input.append(dict(attr_name="long_view", tensor_name="long_view", common=False, dim=1))
        fused_input.append(dict(attr_name="short_view", tensor_name="short_view", common=False, dim=1))
        fused_input.append(dict(attr_name="play_complete", tensor_name="play_complete", common=False, dim=1))
        fused_input.append(dict(attr_name="p_temp", tensor_name="p_temp", common=False, dim=1))
        fused_input.append(dict(attr_name="colossus_time_s", tensor_name="colossus_time_s", common=True, dim=512))
        fused_input.append(dict(attr_name="beam_prob", tensor_name="beam_prob", common=False, dim=1))
        fused_input.append(dict(attr_name="cur_step", tensor_name="cur_step", common=False, dim=1))
        fused_input.append(dict(attr_name="batch_mask", tensor_name="batch_mask", common=False, dim=1))

        print("fused_input: ", fused_input)
        max_batchsize = max(explicit_batchsizes)
        max_num_batches = explicit_max_enqueued_batches
        executor_config = dict(context_per_device=context_per_device)
        model_loader_config = dict(
            rowmajor=True,
            dynamic_shape=True,
            executor_batchsizes=explicit_batchsizes,
            enable_fp16=use_fp16,
            force_input_tensor_fp32=False,
            receive_dnn_model_as_macro_block=True,
        )
        batching_config = dict(
            batch_timeout_micros=0,
            max_batch_size=max_batchsize,
            max_enqueued_batches=max_num_batches,
            enable_pinned_memory=True,
        )

        if use_tvm:
            model_loader_config["type"] = "MioTFExecutedByTVMModelLoader"
            model_loader_config["use_cutlass"] = True
            model_loader_config["tvm_opt_flags"] = (
                "cast_before_square,cutlass,cublas,merge_same_input,mlp_fusion,mha_fusion,concat_mean_fusion,bmm_rewrite_fusion"
            )
            batching_config["batch_task_type"] = "BatchTVMTask"
            if context_per_device > 0:
                batching_config["process_thread_num"] = 2 * context_per_device
            graph = model.graph
        else:
            # raise NotImplementedError("TensorRT is not supported yet")
            model_loader_config["type"] = "MioTFExecutedByTensorFlowModelLoader"
            batching_config["batch_task_type"] = "BasicBatchingTask"
            graph = model.graph

        print("model.outputs: ", model.outputs)

        self.uni_predict_fused(
            embedding_fetchers=[fetch_general_embedding],
            graph=model.graph,
            inputs=fused_input,
            outputs=[
                dict(
                    attr_name=output_prefix + attr_name,
                    tensor_name=tensor_name,
                )
                for attr_name, tensor_name in model.outputs
            ],
            param=model.param,
            queue_prefix=model_config["queue_prefix"],
            key=model_config["queue_prefix"],
            init_from_local_model=False,
            model_loader_config=model_loader_config,
            batching_config=batching_config,
            executor_config=executor_config,
            optimizers=optimizers,
            embedding_manager_type="post_parallel_fetch",
        )
        print("model.outputs: ", model.outputs)

    def fill_full_mask_semantic_id(self):
        self.enrich_attr_by_lua(
            function_for_item="calculate",
            export_item_attr=["init_semantic_id_v2", "init_semantic_id_v2_mask"],
            lua_script="""
                function calculate()
                local token_num = 16
                local semantic_id_v2 = {}
                local semantic_id_v2_mask = {}
                for i = 1, token_num do
                    semantic_id_v2[i] = 0.0
                    semantic_id_v2_mask[i] = 1.0
                end
                return semantic_id_v2, semantic_id_v2_mask
                end
            """
        )

    def retrieve_from_tokens(self):
        self.pack_item_attr(
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "from_item_attr": "token_indices",
                    "to_common_attr": "tokens",
                },
                {
                    "from_item_attr": "token_probs",
                    "to_common_attr": "token_probs",
                }
            ]
        )
        self.enrich_attr_by_lua(
            import_item_attr=["token_indices"],
            export_item_attr=["semantic_id_v2_hash"],
            function_for_item="func",
            lua_script="""
            function func()
                semantic_id_v2_hash = tostring(token_indices[1])
                for i=2,16 do
                    semantic_id_v2_hash = semantic_id_v2_hash .. '.' .. tostring(token_indices[i])
                end
                return util.CityHash64(semantic_id_v2_hash)
            end
            """
        )
        self.fetch_remote_embedding(
            protocol=1,
            colossusdb_embd_model_name="rlj-24q2-norm-exp",
            colossusdb_embd_table_name="code2id_parallel_semantic_id",
            id_converter={"type_name": "plainIdConverter"},
            input_attr_name="semantic_id_v2_hash",
            slot=1,
            output_attr_name="pid",
            query_source_type="item_attr",
            is_raw_data=True,
            raw_data_type="uint64",
            timeout_ms=20,
            size=1,
        )
        self.enrich_attr_by_lua(
            import_item_attr=["pid"],
            export_item_attr=["valid_token_rate"],
            function_for_item="func",
            lua_script="""
            function func()
                return pid ~= nil
            end
            """ 
        )
        self.perflog_attr_value(
            check_point="reco_llada.valid_token_rate",
            item_attrs=["valid_token_rate"],
        )
        self.pack_item_attr(
            target_item={"valid_token_rate": 1},
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "from_item_attr": "pid",
                    "to_common_attr": "gen_pids",
                },
            ]
        )
        self.enrich_attr_by_lua(
            import_common_attr = ["use_ann", "debug_use_ann"],
            function_for_common = "calculate",
            export_common_attr = ["use_ann"],
            lua_script = """
                function calculate()
                    if debug_use_ann ~= nil then
                        return debug_use_ann
                    else
                        return use_ann
                    end
                end
            """
        )
        self.if_("use_ann == 1")
        self.enrich_attr_by_lua(
            import_common_attr = ["i2i_ann_topk", "debug_ann_topk"],
            function_for_common = "calculate",
            export_common_attr = ["i2i_ann_topk"],
            lua_script = """
                function calculate()
                    if debug_ann_topk ~= nil then
                        return debug_ann_topk
                    else
                        return i2i_ann_topk
                    end
                end
            """
        )
        self.delegate_enrich(
            name="ann",
            kess_service="{{rag_service_kess_name}}",
            timeout_ms=200,
            send_item_attrs = [
                {"name": "token_indices", "as": "tokens"}
            ],
            send_common_attrs = [
                {"name": "i2i_ann_topk", "as": "ann_topk"}
            ],
            recv_common_attrs=["ann_pids", "ann_scores"],
            request_type="ann_request"
        )
        self.truncate(size_limit=0)
        self.retrieve_by_common_attr(attr="gen_pids", reason=999)
        self.retrieve_by_common_attr(attr="ann_pids", reason=666)
        self.dispatch_common_attr(
            target_reason=666,
            dispatch_config = [
                {"from_common_attr" : "ann_scores", "to_item_attr" : "ann_score"}
            ]
        )
        self.sort_by(attr="ann_score", target_reason=666)
        self.else_()
        self.truncate(size_limit=0)
        self.retrieve_by_common_attr(attr="gen_pids", reason=999)
        self.end_if_()

        self.if_("request_type ~= 'debug_request'")
        self.deduplicate()
        self.end_if_()

        self.copy_item_meta_info(save_item_key_to_attr="pid")
        return self

    @for_loop(loop_on="infer_steps", loop_value="current_step")
    def infer_step(self):
        self.if_("current_step == 0")
        self.copy_attr(
            attrs=[
                {"from_item": "init_semantic_id_v2", "to_item": "semantic_id_v2"},
                {"from_item": "init_semantic_id_v2_mask", "to_item": "semantic_id_v2_mask"},
            ]
        )
        self.enrich_attr_by_lua(
            function_for_item="calculate",
            import_item_attr=["semantic_id_v2"],
            export_item_attr=["semantic_id_v2_prob", "beam_prob", "batch_mask"],
            lua_script="""
                function calculate(seq)
                    local semantic_id_v2_prob = {}
                    for i = 1, #semantic_id_v2 do
                        semantic_id_v2_prob[i] = 0.0
                    end
                    if seq == 0 then
                        return semantic_id_v2_prob, {0.0}, {1.0}
                    else
                        return semantic_id_v2_prob, {0.0}, {0.0}
                    end
                end
            """
        )
        self.else_()
        self.enrich_attr_by_lua(
            function_for_item="calculate",
            export_item_attr=["batch_mask"],
            lua_script="""
                function calculate()
                    return {1.0}
                end
            """
        )
        self.end_if_()
        self.log_debug_info(
            common_attrs=[
                "adp_effective_view_fix",
                "click",
                "like",
                "follow",
                "forward",
                "comment",
                "long_view",
                "short_view",
                "play_complete",
                "p_topk",
                "p_topp",
                "p_temp",
                "random_remask_flag",
                "current_step",
            ],
            item_attrs=[
                "semantic_id_v2",
                "semantic_id_v2_mask",
                "adp_effective_view_fix",
                "click",
                "like",
                "follow",
                "forward",
                "comment",
                "long_view",
                "short_view",
                "play_complete",
                "p_temp",
                "beam_prob",
                "batch_mask"
            ],
            for_debug_request_only=True,
        )
        self.item_rag()

        self.enrich_attr_by_lua(
            import_common_attr=["current_step"],
            function_for_item="calculate",
            export_item_attr=["cur_step"],
            lua_script="""
                function calculate()
                    local cur_step = {current_step * 1.0}
                    return cur_step
                end
            """,
        )

        self.infer()

        self.log_debug_info(
            log_tag=f"infer_result",
            common_attrs=[
                "current_step",
            ],
            item_attrs=[
                "topk_prob",
                "topk_indices",
                "item_pred_logits",
                "token_prob",
                "item_logit_idx",
                "topk_beam_prob"
            ],
            for_debug_request_only=True,
        )

        self.enrich_attr_by_py(
            function_set=ChooseTokenStrategy,
            py_function=ChooseTokenStrategy.choose_token_beam,
        )

        self.log_debug_info(
            log_tag=f"after_choose_token_beam",
            common_attrs=[
                "current_step",
            ],
            item_attrs=[
                "semantic_id_v2",
                "semantic_id_v2_prob",
                "beam_prob",
            ],
            for_debug_request_only=True,
            respect_sample_logging=False
        )

        self.enrich_attr_by_lua(
            function_for_item="calculate",
            import_item_attr=["semantic_id_v2", "semantic_id_v2_prob"],
            import_common_attr=["current_step"],
            export_item_attr=["token_indices", "token_probs"],
            lua_script="""
                function calculate()
                    local token_indices = {}
                    local token_probs = {}
                    for i = 1, #semantic_id_v2 do
                        table.insert(token_indices, math.ceil(semantic_id_v2[i]))
                        table.insert(token_probs, semantic_id_v2_prob[i])
                    end
                    return token_indices, token_probs
                end
            """
        )

        self.log_debug_info(
            log_tag=f"step_infer_result",
            common_attrs=[
                "current_step",
            ],
            item_attrs=[
                "semantic_id_v2",
                "semantic_id_v2_prob",
                "token_indices",
                "token_probs",
                "topk_indices",
                "topk_prob",
            ],
            for_debug_request_only=True,
            respect_sample_logging=False
        )

    def main(self):
        self.copy_user_meta_info(save_request_type_to_attr="request_type")
        self.set_tab_id(tab_id=TAB_NEBULA)
        self.prepare_infer_params()
        self.prepare_fake_item()
        self.prepare_user_info()
        self.extract_feature()
        self.fill_full_mask_semantic_id()

        self.enrich_attr_by_lua(
            import_common_attr=["infer_step"],
            export_common_attr=["infer_steps"],
            function_for_common="calculate",
            lua_script="""
                function calculate()
                    local infer_steps = {}
                    for i = 1, infer_step do
                        table.insert(infer_steps, i - 1)
                    end
                    return infer_steps
                end
            """,
        )
        self.infer_step()

        self.enrich_attr_by_lua(
            function_for_item="calculate",
            import_item_attr=["semantic_id_v2"],
            export_item_attr=["token_indices"],
            lua_script="""
                function calculate()
                    local token_indices = {}
                    for i = 1, #semantic_id_v2 do
                        table.insert(token_indices, math.ceil(semantic_id_v2[i]))
                    end
                    return token_indices
                end
            """,
        )
        self.log_debug_info(
            log_tag="final_output",
            item_attrs=["token_indices", "token_probs"],
            for_debug_request_only=True,
            respect_sample_logging=False
        )
        self.retrieve_from_tokens()
        return self


predict_for_all = PredictServerFlow(name="predict_for_all").main()

eval_flow = HitRatePerfFlow(name="hit_rate_perf").hit_rate_perf()
eval_reward_flow = EvalFlow(name="eval_reward_flow").eval_pointwise_reward(kess_name)

service = LeafService(
    kess_name=kess_name,  # 该 kess 不生效，由 krp 替换
    item_attrs_from_request=["reco_photo_info_str", "living"],
    common_attrs_from_request=[
        "is_debug",
        "user_info_str",
        "user",
        "source_photo_id",
        "page_common",
        "colossus_response_str",
        "tab_id",
        "is_tnu",
        "gen_item_num"
    ],
)

service.common_attrs_from_request += [
    "adp_effective_view_fix",
    "click",
    "like",
    "follow",
    "forward",
    "comment",
    "long_view",
    "short_view",
    "play_complete",
    "p_topk",
    "p_topp",
    "p_temp",
    "random_remask_flag",
    "eval_pos_photo_id_list",
    "debug_ann_topk",
    "debug_use_ann"
]
service.return_common_attrs(["i2i_ann_topk", "tokens", "token_probs"])
service.return_item_attrs(["pid", "ann_score", "src_item"])

# 不要修改
service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
service.CHECK_UNUSED_ATTR = False

service.add_leaf_flows(leaf_flows=[predict_for_all, eval_flow], request_type="predict_for_all")
service.add_leaf_flows(leaf_flows=[predict_for_all], request_type="default", as_default=True)
service.add_leaf_flows(leaf_flows=[predict_for_all], request_type="debug_request")
service.add_leaf_flows(leaf_flows=[predict_for_all, eval_flow, eval_reward_flow], request_type="eval_request")

if __name__ == "__main__":
    out_file = str(__file__).replace("py", "json")
    service.build(output_file=out_file)
