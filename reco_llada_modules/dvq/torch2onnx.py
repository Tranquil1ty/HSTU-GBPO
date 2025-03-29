import os
import torch, onnx
from torch import nn
from argparse import ArgumentParser
import yaml
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from onnxruntime import InferenceSession, SessionOptions

from vqvae import VQVAE


# 首先定义一个 module 调用 model.infer 接口

class VQVAEWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens,):
        bs, n_tokens = tokens.shape
        embs, _ = self.model.infer(tokens)
        return {
            "embs": embs
        }

def build_pytorch_param_index(pytorch_params):
    """
    构建 PyTorch 参数的索引，按形状和转置形状分类
    """
    param_index = dict()
    for name, param in pytorch_params.items():
        param_array = param.cpu().numpy()
        shape = tuple(param_array.shape)

        # 记录原始形状
        if shape not in param_index:
            param_index[shape] = []
        param_index[shape].append((name, param_array))

    return param_index

def match_parameters(onnx_name, onnx_param, pytorch_param_index):
    """
    匹配单个 ONNX 参数与 PyTorch 参数。
    """
    onnx_param_array = np.frombuffer(onnx_param.raw_data, dtype=np.float32).reshape(tuple(onnx_param.dims))
    onnx_shape = tuple(onnx_param_array.shape)

    # 尝试直接匹配形状
    if onnx_shape in pytorch_param_index:
        for name, param_array in pytorch_param_index[onnx_shape]:
            if np.allclose(param_array, onnx_param_array):
                return onnx_name, (name, param_array.shape)  # 匹配成功

    # 尝试匹配转置形状
    transposed_shape = onnx_shape[::-1]
    if transposed_shape in pytorch_param_index:
        for name, param_array in pytorch_param_index[transposed_shape]:
            if (
                np.allclose(param_array.swapaxes(0, 1), onnx_param_array)
            ):
                return onnx_name, (name, param_array.shape, "need transpose")  # 匹配成功，需转置

    # 未匹配
    return onnx_name, None

def validate_onnx_pytorch_mapping(onnx_params, pytorch_params):
    """
    验证 ONNX 模型和 PyTorch 模型之间的参数映射关系。
    """
    # 构建 PyTorch 参数索引
    pytorch_param_index = build_pytorch_param_index(pytorch_params)

    # 并行化匹配
    name_mapping = dict()
    unmatched = []
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda item: match_parameters(item[0], item[1], pytorch_param_index),
                onnx_params.items(),
            )
        )

    # 处理结果
    for onnx_name, mapping in results:
        if mapping is not None:
            name_mapping[onnx_name] = mapping
        else:
            unmatched.append(onnx_name)

    # 打印未匹配的参数
    if unmatched:
        print(f"未匹配的参数: {unmatched}")

    # 打印最终映射
    print(f"\t[ONNX -> PyTorch] 参数映射[{len(name_mapping.keys())}]:\n {name_mapping}")
    return name_mapping

def _dump_params(pt_model, origin_model):
    pytorch_params: dict = pt_model.state_dict()
    onnx_params = {init.name: init for init in origin_model.graph.initializer}
    print('x' * 100)
    print('onnx model param keys: ', onnx_params.keys())
    print('onnx model param keys: ', len(onnx_params.keys()), ', unique keys: ', len(set(onnx_params.keys())))
    print('pytorch model param keys: ', len(pytorch_params.keys()), ', unique keys: ', len(set(pytorch_params.keys())))
    print('x' * 100)
    name_mapping = validate_onnx_pytorch_mapping(onnx_params, pytorch_params)

    return name_mapping

def __dump_params(pt_model, onnx_model):
    pytorch_params: dict = pt_model.state_dict()
    onnx_params = {init.name: init for init in onnx_model.graph.initializer}
   
  	# 保存 onnx-> pytorch 模型的参数映射关系
    name_mapping = {}
    for onnx_name, onnx_param in onnx_params.items():
        assert onnx_name in simp_params_name, f"{onnx_name} not in simplified onnx model."
        onnx_param_array = np.frombuffer(onnx_param.raw_data, dtype=np.float32).reshape(
            tuple(onnx_param.dims)
        )
        # 暴力匹配
        is_matched_ = False
        for name, param in pytorch_params.items():
            param_array = param.cpu().numpy()
            if (
                not is_matched_
                and len(param_array.shape) == 2  # 目前用到的权重最大 dims 最大为 2
                and param_array.shape[::-1] == onnx_param_array.shape
            ):
                if np.allclose(param_array.swapaxes(0, 1), onnx_param_array):
                    name_mapping[onnx_name] = (name, param_array.shape, "need transpose")
                    is_matched_ = True
            if not is_matched_ and param_array.shape == onnx_param_array.shape:
                if np.allclose(param_array, onnx_param_array):
                    name_mapping[onnx_name] = (name, param_array.shape)
                    is_matched_ = True
        if not is_matched_:
            print(f"{onnx_name} not matched")
    print(f"\t[onnx -> pytorch] name mapping:\n {name_mapping}")
    return name_mapping

def torch2onnx(args):
    model = VQVAE.load_from_checkpoint(args.ckpt, args=args)
    
    # 导出模型结构（带参数）
    vqvae_wrapper = VQVAEWrapper(model).eval().cuda()
    onnx_file_name = os.path.join(args.onnx_out_dir, "vqvae.onnx")
    batch = 1
    n_tokens = 16
    tokens = torch.ones([batch, n_tokens], dtype=torch.float32).cuda()

    in_names = ["tokens"]
    out_names = ["embs"]
    input_datas = (tokens)

    torch.onnx.export(
        vqvae_wrapper,
        input_datas,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes={
            "tokens": {0: "batchsize"},
            # list value: automatic names
            "embs": {0: "batchsize"},
            "z": {0: "batchsize"},
        },
    )

    # 仅导出模型结构
    # onnx.save(
    # onnx_model,
    # "llama_onnx_structure.onnx",
    # save_as_external_data=True,
    # all_tensors_to_one_file=True,
    # location="llama_onnx_structure_data",
    # )

    # 输出配置文件
    dnn_model_config = dict()
    dnn_model_config["output_names"] = out_names
    
    # sparse 参数 （略）
    # dnn_model_config["embedding"] = dict()
    # dnn_model_config["embedding"]["slots_config"] = [
    #     {
    #         "common": True,
    #         "dim": config.hidden_units, 
    #         "dtype": "mio_int16",
    #         "expand": config.seq_len_photo_input,
    #         "input_name": "input_embeds",
    #         "slots": "1",
    #     },
    # ]

    # dense 参数
    onnx_model = onnx.load(onnx_file_name)
    onnx2pytorch_param_mapping = _dump_params(vqvae_wrapper, onnx_model)
    dnn_model_config["param"] = []
    for k, v in onnx2pytorch_param_mapping.items():
        dnn_model_config["param"].append({
            'th_tensor_name': v[0],
            'rown': v[1][0],
            'coln': 1 if len(v[1]) < 2 else v[1][1],
            'name': k,
            'need_transpose': False if len(v) < 3 else True,
        })

    with open(f'{args.onnx_out_dir}/dnn_model.yaml', 'w') as file:
        yaml.dump(dnn_model_config, file, default_flow_style=False, allow_unicode=True)
        print("YAML 文件已成功导出")
    

    # dump parameters to npz
    state_dict = model.state_dict()
    numpy_dict = {name: param.cpu().numpy() for name, param in state_dict.items()}
    print(numpy_dict.keys())
    np.savez("model_weights.npz", **numpy_dict)


def validate(args):

    model = VQVAE.load_from_checkpoint(args.ckpt, args=args)
    model.eval().cuda()

    tokens = torch.tensor([
        [487, 429, 97, 100, 329, 327, 368, 169, 397, 97, 103, 474, 73, 312, 450, 511],
        [228, 330, 427, 5, 314, 342, 364, 362, 317, 177, 140, 394, 489, 126, 17, 286]
        # [0] * 16
        ], dtype=torch.float32).cuda()
    x_hat, z = model.infer(tokens)
    print(x_hat)
    print(z)

    onnx_file_name = os.path.join(args.onnx_out_dir, "vqvae.onnx")
    # onnx_model = onnx.load(onnx_file_name)
    session = InferenceSession(onnx_file_name, providers=[
            "CUDAExecutionProvider",  # 优先使用 CUDA
            "CPUExecutionProvider"   # 如果 CUDA 不可用，则使用 CPU
        ])
    onnx_inputs = {}
    onnx_inputs["tokens"] = tokens.cpu().numpy().tolist()
    print(onnx_inputs)
    # Compute outputs from the ONNX model
    onnx_outputs = session.run(None, onnx_inputs)
    print(onnx_outputs[0])
    print(onnx_outputs[1])

def main():

    parser = ArgumentParser()
    parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
    parser.add_argument("--enc_dec_flavor", type=str, default='deepmind', choices=['deepmind', 'openai'])
    parser.add_argument("--loss_flavor", type=str, default='l2', choices=['l2', 'logit_laplace'])
    parser.add_argument("--input_dim", type=int, default=128, help="input dim")
    parser.add_argument("--num_embeddings", type=int, default=512, help="vocabulary size; number of possible discrete states")
    parser.add_argument("--embedding_dim", type=int, default=32, help="size of the vector of the embedding of each discrete token")
    parser.add_argument("--n_hid", type=int, default=512, help="number of channels controlling the size of the model")
    parser.add_argument("--n_token", type=int, default=16, help="number of qunatized tokens")
    parser.add_argument("--data_dir", type=str, default='viewfs://hadoop-lt-cluster/home/reco_kaiworks/dw/reco_kaiworks.db/rlj_semantic_id_training_samples/p_date=20250311')
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='/pub/renlejian/repo/deep-vector-quantization/dvq/lightning_logs/version_18/checkpoints/epoch=38-step=241917.ckpt')
    parser.add_argument("--onnx_out_dir", type=str, default='/pub/renlejian/repo/deep-vector-quantization/dvq/deploy/')
    parser.add_argument("--opset", required=False, type=int, default=17)

    
    args = parser.parse_args()

    torch2onnx(args)
    validate(args)

    
if __name__ == "__main__":
    main()