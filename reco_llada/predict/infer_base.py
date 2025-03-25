import yaml
import collections
from typing import Dict, List
import os
import base64
import sys


ModelConfig = collections.namedtuple(
    "ModelConfig",
    [
        "graph",
        "outputs",
        "slots_config",
        "extra_inputs",
        "param",
        "feature_list",
        "feature_to_slots",
    ],
)


photo_store_config = dict(
    use_new_index=True,
    distributed_index="reco.distributedIndex.recoExploreFastPhotoStoreConfigNuma",
)

extra_reco_photo_info_attrs = [
    dict(path="cascade_pctr_index", name="cascade_pctr_index"),
    dict(path="cascade_plvtr_index", name="cascade_plvtr_index"),
    dict(path="cascade_pvtr_index", name="cascade_pvtr_index"),
    dict(path="cascade_pltr_index", name="cascade_pltr_index"),
    dict(path="cascade_pwtr_index", name="cascade_pwtr_index"),
    dict(path="cascade_pftr_index", name="cascade_pftr_index"),
]


def kuiba_list_converter_config_generator(limit=None):
    return {
        "converter": "list",
        "converter_args": {
            "reversed": False,
            "enable_filter": False,
            **({"limit": limit} if limit is not None else {}),
        },
    }


kuiba_list_converter_config = kuiba_list_converter_config_generator()
kuiba_list_converter_config_limit800 = kuiba_list_converter_config_generator(limit=800)
kuiba_list_converter_config_limit400 = kuiba_list_converter_config_generator(limit=400)
kuiba_list_converter_config_limit300 = kuiba_list_converter_config_generator(limit=300)
kuiba_list_converter_config_limit200 = kuiba_list_converter_config_generator(limit=200)
kuiba_list_converter_config_limit100 = kuiba_list_converter_config_generator(limit=100)
kuiba_list_converter_config_limit120 = kuiba_list_converter_config_generator(limit=120)
kuiba_list_converter_config_limit50 = kuiba_list_converter_config_generator(limit=50)
kuiba_list_converter_config_limit40 = kuiba_list_converter_config_generator(limit=40)
kuiba_list_converter_config_limit20 = kuiba_list_converter_config_generator(limit=20)
kuiba_list_converter_config_limit10 = kuiba_list_converter_config_generator(limit=10)
kuiba_list_converter_config_limit7 = kuiba_list_converter_config_generator(limit=7)
kuiba_list_converter_config_limit6 = kuiba_list_converter_config_generator(limit=6)
kuiba_list_converter_config_limit2 = kuiba_list_converter_config_generator(limit=2)
kuiba_list_converter_config_for_aid_sim = kuiba_list_converter_config_limit50
kuiba_list_converter_config_for_dual_sim = kuiba_list_converter_config_limit200
kuiba_list_converter_config_for_dual_sim_cross = kuiba_list_converter_config_limit200
kuiba_list_converter_config_limit_n = (
    lambda limit: kuiba_list_converter_config_generator(limit)
)
kuiba_id_converter_config = {"converter": "id"}


def load_feature_list_sign(filename):
    ret = set()
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            for field in line.strip().split(","):
                parts = field.strip().split("=")
                assert len(parts) == 2, "Unsupported format: " + line.strip()

                if parts[0].strip() != "class":
                    # ignore unknown field
                    continue

                ret.add(parts[1].strip())
    return list(sorted(ret))


def load_feature_list_with_slots(filename: str) -> Dict[str, List[int]]:
    format_example = "# slot = 731 + 16 = 747\nclass = ExtractSignUserLowActive\n"
    format_example += (
        "# slot = 815, 825, 835\nclass = ExtractSignAuthorIdUniversal_799\n"
    )
    format_example += "# slot = 567\nclass = ExtractSignCascadeXctrV3\n"
    format_example += "# 用户省份和城市单边特征，slot = 166 + 16 = 182\nclass = ExtractSignUserProvCity\n"

    ret = {}
    need_class_line = False
    slot_id_list = []
    with open(filename) as f:
        for line in f:
            orig_line = line
            line = line.replace("\n", "").replace(" ", "")
            if not line:
                continue
            err_msg = (
                "file: "
                + filename
                + "; cur line: "
                + orig_line
                + "\nbad1 format. e.g: "
                + format_example
            )
            if line.startswith("#"):
                if len(line.split("slot=")) == 1:
                    slot_id_list = []
                    need_class_line = False
                    continue
                s1 = line.split("=")
                if (
                    need_class_line
                    or not (len(s1) != 2 or len(s1) != 3)
                    or s1[0][-4:] != "slot"
                ):
                    raise RuntimeError(err_msg)
                slot_list_str = s1[-1]
                s2 = slot_list_str.split(",")
                if len(s2) < 1:
                    raise RuntimeError(err_msg)
                slot_id_list = []
                for k in s2:
                    if not k.isdigit():
                        raise RuntimeError(k + " is not digit; " + err_msg)
                    slot_id_list.append(int(k))
                need_class_line = True
                continue
            if not need_class_line:
                raise RuntimeError(err_msg)

            class_name_split = line.strip().split(",")
            if len(class_name_split) != 1:
                raise RuntimeError("report to lixinhua03 !!!\n" + err_msg)
            class_name_list = class_name_split[0]
            parts = class_name_list.strip().split("=")
            if len(parts) != 2:
                raise RuntimeError("Unsupported format: " + err_msg)

            if parts[0].strip() != "class":
                raise RuntimeError(err_msg)
                continue
            class_name = parts[1].strip()
            ret[class_name] = slot_id_list
            if len(slot_id_list) == 0:
                raise RuntimeError("slot number is zero: " + err_msg)
            slot_id_list = []
            need_class_line = False
    return ret


def load_model(model_dir):
    with open(os.path.join(model_dir, "dnn_model.yaml")) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(model_dir, "graph.pb"), "rb") as f:
        base64_graph = base64.b64encode(f.read()).decode("ascii")
        graph = "base64://" + base64_graph

    feature_to_slots = load_feature_list_with_slots(
        os.path.join(model_dir, "feature_list_sign.txt")
    )
    feature_list = load_feature_list_sign(
        os.path.join(model_dir, "feature_list_sign.txt")
    )

    graph_tensor_mapping = dnn_model["graph_tensor_mapping"]
    extra_preds = dnn_model["extra_preds"].split(" ")
    q_names = dnn_model["q_names"].split(" ")
    assert len(extra_preds) == len(q_names)
    outputs = [
        (extra_pred, graph_tensor_mapping[q_name])
        for extra_pred, q_name in zip(extra_preds, q_names)
    ]
    param = [param for param in dnn_model["param"] if param.get("send_to_online", True)]

    slots_config = dnn_model["embedding"]["slots_config"]
    for sc in slots_config:
        if "dtype" in sc:
            sc["tensor_dtype"] = sc["dtype"]
            del sc["dtype"]

    extra_inputs = dnn_model["vec_input"]

    return ModelConfig(
        graph,
        outputs,
        slots_config,
        extra_inputs,
        param,
        feature_list,
        feature_to_slots,
    )

def extract_input_from_slot_config(c):
  ret = dict(
    attr_name=c["input_name"],
    tensor_name=c["input_name"],
    common=c.get("common", False),
    dim=len(str(c["slots"]).split(" ")) * c["dim"] * c.get("expand", 1) + (1 if c.get("sized", False) else 0),
  )

  if c.get("compress_group") == "USER":
    ret["compress_group"] = "USER"

  return ret