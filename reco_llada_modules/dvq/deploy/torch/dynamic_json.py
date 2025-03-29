#!/usr/bin/env python3
# coding=utf-8

import os, sys
import base64
import collections
import yaml


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

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../dragon/tools/pypi/"))
# sys.path.append('/home/root/dragon_kbuild/dragon/tools/pypi/')

config_root_dir = os.path.join(os.path.dirname(__file__))

colossusdb_embd_service_name = "rlj-24q2-norm-exp"
colossusdb_embd_table_name= "GRMListGen09BGRPO"
service_name= "grpc_semantic_id_decoder"
queue_prefix = "semantic_id_decoder"   #

model_key = "semantic_id_decoder"

EOL="\n"



ModelConfig = collections.namedtuple(
    "ModelConfig",
    [
        "graph",
        "outputs",
        "param",
    ],
)

def load_onnx_model():
    model_dir = os.path.join(os.path.dirname(__file__))
   # 输入输出定义配置文件
    with open(os.path.join(model_dir, "dnn_model.yaml")) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)
        print('dnn_model.yaml'.center(100, '*'))
        print(dnn_model)
    
    outputs = [(x, x) for x in dnn_model['output_names']]
    print(outputs)
    param = [
        param
        for param in dnn_model["param"]
        if param.get("send_to_online", True)
    ]

    # 模型权重
    with open(os.path.join(model_dir, "vqvae.onnx"), "rb") as f:
        # graph = '''          "graph": "base64://''' + base64.b64encode(onnx_model_def.SerializeToString()).decode('ascii') + '''",'''
        base64_graph = base64.b64encode(f.read()).decode("ascii")
        graph = "base64://" + base64_graph

    return ModelConfig(
        graph,
        outputs,
        param
    )

model_config = load_onnx_model()


mock_slots_config = []
# for c in model_config.slots_config:
sc = dict()
sc['input_name'] = "inputs_ids_embeddings"
sc['slots'] = '1'
sc['dtype'] = 'mio_int16'
#sc['dtype'] = 'scale_int8'
sc['expand'] = 1
sc['dim'] = 1024
sc['common'] = True
mock_slots_config.append(sc)

class DecodeFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, EmbedCalcApiMixin, GsuApiMixin, PDNApiMixin, CofeaApiMixin, UniPredictV2ApiMixin, EmbeddingApiMixin):

    def decode(self, **kwargs):

        return (
            self
            .cast_attr_type(
                attr_type_cast_configs=[
                    {
                    "to_type": "double",
                    "from_item_attr": "tokens",
                    "to_item_attr": "tokens_double"
                    },
                ]
            )
            .gen_common_attr_by_lua(
                attr_map={
                    "common_slots": "{1}",
                    "common_parameters": "{1152921504730303765}",
                }
            )           
            .uni_predict_fused(
                debug_tensor=True,
                using_onnx_style_name=True,
                graph=model_config.graph,
                optimizers=["FuseMioVariableCast"],
                key=model_key,
                queue_prefix=queue_prefix,
                inputs=[
                    {
                        "attr_name": "tokens_double",
                        "tensor_name": "tokens",
                        "dim": 16
                    }
                ],
                outputs=[dict(attr_name=attr_name, tensor_name=tensor_name) for attr_name, tensor_name in model_config.outputs],
                param=model_config.param,
                model_loader_config=dict(rowmajor=True,
                                        type="MioTFExecutedByOnnxTRTModelLoader",
                                        # implicit_batch=False,
                                        dynamic_shape=True,
                                        executor_batchsizes=[1024],  # user 单边预估，batch_size 为 1
                                        receive_dnn_model_as_macro_block=False,
                                        enable_xla=False,
                                        enable_bf16=False,
                                        force_input_tensor_fp32=True,
                                        # enable_fp16=False,
                                        ),
                batching_config=dict(batch_timeout_micros=0,
                                    max_batch_size=1024,
                                    max_enqueued_batches=1,
                                    batch_task_type="BasicBatchingTask"),
                executor_config=dict(intra_op_parallelism_threads_num=4,
                                    inter_op_parallelism_threads_num=4,
                                    context_per_device=12,
                                    ),
                embedding_fetchers=[dict(fetcher_type="ColossusdbEmbeddingServerFetcher",
                                    colossusdb_embd_service_name=colossusdb_embd_service_name,
                                    colossusdb_embd_table_name=colossusdb_embd_table_name,
                                    common_slots_inputs=["common_slots"],
                                    common_parameters_inputs=["common_parameters"],
                                    slots_config=mock_slots_config)]
            )
            .log_debug_info(
                item_attrs = ["tokens", "embs", "tokens_double"], 
                for_debug_request_only=True, 
                respect_sample_logging=False
            )
        )

decode_flow = DecodeFlow(name="decode_flow").decode()

kess_name = service_name
print(f"kess name: {kess_name}")


service = LeafService(
    kess_name=kess_name,
    # common_attrs_from_request=["tokens"],
    item_attrs_from_request=["tokens"],
    index_source=IndexSource.LOCAL_ATTR_INDEX,
)

service.AUTO_INJECT_ITEM_ATTR = False
service.CHECK_UNUSED_ATTR = False

# service.IGNORE_NO_SOURCE_ATTR=["browsed_pids", "session_photo_id_list", 'session_tag_list',]
# service.return_common_attrs(attrs=["embs", "z"])
service.return_item_attrs(attrs=["embs"])
# returned_item_attrs = ["embs"]
# service.return_item_attrs(attrs=returned_item_attrs)

service.add_leaf_flows(request_type="default", leaf_flows=[decode_flow, ], as_default=True)
service.add_leaf_flows(request_type="decode_request", leaf_flows=[decode_flow, ])

service.build(output_file=__file__.replace(".py", ".json"))
