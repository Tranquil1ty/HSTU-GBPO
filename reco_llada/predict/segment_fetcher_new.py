slot_as_attr_name_prefix = "gsu_slots_attr_"
USE_SELECT_SIGN_REPLACE_LUA = True
init_from_local_model = False

# -->> 为了支持 memcpy, memcpy 存在一个问题，就是同一个 sign 属于两个 slot 时，可能存在问题
def SortSlotConfigByExpand(one_fetcher):
    tmp_fetcher = one_fetcher
    slots_config = one_fetcher["slots_config"]
    tmp_fetcher["slots_config"] = sorted(slots_config, key=lambda x: (x["expand"] if "expand" in x else 1))
    return tmp_fetcher

def GenSegmentConfigSaAN(fetch_config, slot_as_attr_name_list, slots_inputs, attr_preix = slot_as_attr_name_prefix, one_piece_limit = 7):
    dim_to_slot = dict()
    for k in fetch_config["slots_config"]:
        slots = k["slots"].split()
        hit_num = 0
        for s in slots:
            if int(s) in slot_as_attr_name_list:
                hit_num = hit_num + 1
                dim_to_slot[int(k["dim"]) * 100000 + int(s)] = int(s)
        if hit_num > 0 and hit_num != len(slots):
            raise RuntimeError("ERROR: part include, input_name: " + k["input_name"] + "; slots: " + k["slots"])
    dim_to_slot_sorted = sorted(dim_to_slot.items(), reverse = True)
    slots_list = [i[1] for i in dim_to_slot_sorted]
    piece_num = int((len(slots_list) + one_piece_limit - 1) / one_piece_limit)
    tmp_segment_config = []
    for loop in range(piece_num):
        tmp_dict = dict()
        tmp_dict["slots_inputs"] = slots_inputs # ["gsu_slots"]
        tmp_dict["slots"] = slots_list[loop::piece_num]
        tmp_dict["slot_as_attr_name"] = True
        tmp_dict["slot_as_attr_name_prefix"] = attr_preix
        tmp_segment_config.append(tmp_dict)
    return tmp_segment_config

def RemoveElemByIdx(data_list, idx_list):
    res_list = []
    for i in range(len(data_list)):
        if i not in idx_list:
            res_list.append(data_list[i])
    return res_list

def SegmentFetcher(fetch_config, segment_config):
    fetch_config_bak = fetch_config

    rebuild_attr_list = ["common_slots_inputs", "common_parameters_inputs", "slots_inputs", "parameters_inputs", "slots_config"]
    res_config = []
    common_seg_idx = []
    seg_idx = []
    slots_config_seg_idx = []
    for one_segment in segment_config:
        tmp_common_seg_idx = []
        tmp_seg_idx = []
        tmp_slots_config_seg_idx = []
        if "slots_inputs" in one_segment:
            for k in one_segment["slots_inputs"]:
                if "slots_inputs" in fetch_config and k in fetch_config["slots_inputs"]:
                    tmp_seg_idx.append(fetch_config["slots_inputs"].index(k))
                else:
                    raise RuntimeError('bad slots_inputs config' + str(one_segment["slots_inputs"]))
        if "common_slots_inputs" in one_segment:
            for k in one_segment["common_slots_inputs"]:
                if "common_slots_inputs" in fetch_config and k in fetch_config["common_slots_inputs"]:
                    tmp_common_seg_idx.append(fetch_config["common_slots_inputs"].index(k))
                else:
                    raise RuntimeError('bad common_slots_inputs config' + str(one_segment["common_slots_inputs"]))
                
        for k in fetch_config["slots_config"]:
            slots = k["slots"].split()
            hit_num = 0
            for s in slots:
                if int(s) in one_segment["slots"]:
                    hit_num = hit_num + 1
            if hit_num == len(slots):
                tmp_slots_config_seg_idx.append(fetch_config["slots_config"].index(k))
                continue
            if hit_num > 0:
                raise RuntimeError("ERROR: part include, input_name: " + k["input_name"] + "; slots: " + k["slots"])
        one_res = dict()
        if "slot_as_attr_name" in one_segment and one_segment['slot_as_attr_name'] == True:
            slot_as_attr_name_prefix = "" if "slot_as_attr_name_prefix" not in one_segment else one_segment["slot_as_attr_name_prefix"]
            if len(tmp_common_seg_idx) > 0 and len(tmp_seg_idx) > 0:
                raise RuntimeError('common and item both set slot_as_attr_name not support')
            tslot = one_segment["slots"] # here INT (important)
            tsign = [slot_as_attr_name_prefix + str(k) for k in one_segment["slots"]]
            one_res["slot_as_attr_name"] = True
            if len(tmp_common_seg_idx) > 0:
                one_res["common_slots_inputs"] = tslot
                one_res["common_parameters_inputs"] = tsign
                for k in range(len(tmp_common_seg_idx)):
                    idx = tmp_common_seg_idx[k]
                    common_seg_idx.append(idx)
            else:
                one_res["slots_inputs"] = tslot
                one_res["parameters_inputs"] = tsign
                for k in range(len(tmp_seg_idx)):
                    idx = tmp_seg_idx[k]
                    seg_idx.append(idx)
        else:
            if len(tmp_common_seg_idx) > 0:
                one_res["common_slots_inputs"] = []
                one_res["common_parameters_inputs"] = []
                for k in range(len(tmp_common_seg_idx)):
                    idx = tmp_common_seg_idx[k]
                    common_seg_idx.append(idx)
                    one_res["common_slots_inputs"].append(fetch_config["common_slots_inputs"][idx])
                    one_res["common_parameters_inputs"].append(fetch_config["common_parameters_inputs"][idx])
            if len(tmp_seg_idx) > 0:
                one_res["slots_inputs"] = []
                one_res["parameters_inputs"] = []
                for k in range(len(tmp_seg_idx)):
                    idx = tmp_seg_idx[k]
                    seg_idx.append(idx)
                    one_res["slots_inputs"].append(fetch_config["slots_inputs"][idx])
                    one_res["parameters_inputs"].append(fetch_config["parameters_inputs"][idx])
        if len(tmp_slots_config_seg_idx) > 0:
            one_res["slots_config"] = []
            for k in range(len(tmp_slots_config_seg_idx)):
                idx = tmp_slots_config_seg_idx[k]
                slots_config_seg_idx.append(idx)
                one_res["slots_config"].append(fetch_config["slots_config"][idx])
        
        for k,v in fetch_config.items():
            if k not in rebuild_attr_list:
                one_res[k] = v

        res_config.append(SortSlotConfigByExpand(one_res))

    if "common_slots_inputs" in fetch_config_bak:
        fetch_config_bak["common_slots_inputs"] = RemoveElemByIdx(fetch_config_bak["common_slots_inputs"], common_seg_idx)
        fetch_config_bak["common_parameters_inputs"] = RemoveElemByIdx(fetch_config_bak["common_parameters_inputs"], common_seg_idx)
    if "slots_inputs" in fetch_config_bak:
        fetch_config_bak["slots_inputs"] = RemoveElemByIdx(fetch_config_bak["slots_inputs"], seg_idx)
        fetch_config_bak["parameters_inputs"] = RemoveElemByIdx(fetch_config_bak["parameters_inputs"], seg_idx)
    fetch_config_bak["slots_config"] = RemoveElemByIdx(fetch_config_bak["slots_config"], slots_config_seg_idx)
    res_config.append(SortSlotConfigByExpand(fetch_config_bak))
    return res_config

