from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.common.common_api_mixin import CommonApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from infer_base import (
    photo_store_config, extra_reco_photo_info_attrs,
    load_model, extract_input_from_slot_config, kuiba_list_converter_config_limit50
)
from segment_fetcher_new import (
    GenSegmentConfigSaAN,
    USE_SELECT_SIGN_REPLACE_LUA
)

TAB_NEBULA = 30000

model_config = dict(
    model_config=load_model("./config"),
    colossusdb_embd_service_name="reco_llada",
    colossusdb_embd_table_name="emb_reco_llada",
    embedding_shard_num=8,
    queue_prefix="reco_llada",
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
    use_fp16=True,
    use_fused_op=True,
    context_per_device=12,
    executor_per_flavor=12,
    optimizers=[
        #'MatmulBiasaddReluFusion',
        # 'CascadeMatMulFusion',
    ],
    explicit_batchsizes=[32],
    explicit_max_enqueued_batches=8,
    use_tvm=True
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

class PredictServerFlow(LeafFlow, CommonApiMixin, KuibaApiMixin, MioApiMixin, UniPredictApiMixin):

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

    def prepare_user_info(self):
        self.parse_protobuf_from_string(
            is_common_attr=True,
            input_attr="user_info_str",
            output_attr="user_info",
            class_name="ks::reco::UserInfo",
        )
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

    def extract_feature(self):
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
        self.copy_features(
            common_slots_input="common_slots",
            common_parameters_input="common_signs",
            common_slots_mapping=model_config["common_slots_mapping"],
            common_slots_output="copy_common_slots",
            common_parameters_output="copy_common_signs",
            list_enable_filter=False,
        )
        self.log_debug_info(
            for_debug_request_only=False,
            common_attrs=[
                'copy_common_slots',
                'copy_common_signs',
            ]
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

    def infer(self):
        colossusdb_embd_service_name = model_config["colossusdb_embd_service_name"]
        colossusdb_embd_table_name = model_config["colossusdb_embd_table_name"]
        common_slots_inputs = [
            "common_slots",
            "copy_common_slots",
            "kuiba_common_slots",
        ]
        common_parameters_inputs = [
            "common_signs",
            "copy_common_signs",
            "kuiba_common_signs",
        ]
        embedding_dtype = model_config["embedding_dtype"]
        model = model_config["model_config"]
        use_fp16 = model_config.get("use_fp16", False)
        explicit_batchsizes = model_config.get("explicit_batchsizes", [32])
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
            common_slots_inputs=common_slots_inputs,
            common_parameters_inputs=common_parameters_inputs,
            slots_config=[
                dict(dtype=embedding_dtype, **sc) for sc in model.slots_config
            ],
            max_signs_per_request=500,
            fetcher_type="ColossusdbEmbeddingServerFetcher",
        )
        fused_input = [extract_input_from_slot_config(c) for c in model.slots_config]
        print("fused_input: ", fused_input)
        max_batchsize = max(explicit_batchsizes)
        max_num_batches = explicit_max_enqueued_batches
        executor_config = dict(context_per_device=context_per_device)
        model_loader_config = dict(
            rowmajor=True,
            implicit_batch=False,
            executor_batchsizes=explicit_batchsizes,
            enable_fp16=use_fp16,
            force_input_tensor_fp32=False,
            receive_dnn_model_as_macro_block=True,
        )
        batching_config = dict(
            batch_timeout_micros=40000,
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
            raise NotImplementedError("TensorRT is not supported yet")

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

    def main(self):
        self.set_tab_id(tab_id=TAB_NEBULA)
        self.prepare_user_info()
        self.extract_feature()
        self.infer()
        return self


predict_for_all = PredictServerFlow(name="predict_for_all").main()

service = LeafService(
    kess_name="grpc_RecoLlada",  # 该 kess 不生效，由 krp 替换
    item_attrs_from_request=["reco_photo_info_str", "living"],
    common_attrs_from_request=[
        "user_info_str",
        "source_photo_id",
        "page_common",
        "colossus_response_str",
        "tab_id",
        "is_tnu",
    ],
)

service.return_item_attrs(["adp_evtr"])

# 不要修改
service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False

service.add_leaf_flows(leaf_flows=[predict_for_all], request_type="predict_for_all")

if __name__ == "__main__":
    out_file = str(__file__).replace("py", "json")
    service.build(output_file=out_file)
