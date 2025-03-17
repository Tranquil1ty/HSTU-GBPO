import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    pos_emb_type: str = "rope"
    rope_theta: float = 500000

    max_batch_size: int = 32
    #max_seq_len: int = 2048

    is_varlen: bool = True
    last_layer_norm: bool = True

    lora_alpha: float = 1
    lora_dropout: float = 0.0
    lora_r: int = 0

    num_experts: int = 1
    moe_top_k: int = 1
    is_perf_moe_metric: bool = False


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass


def silu(x):
    return x * tf.nn.sigmoid(x)


class Dense(Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        use_bias: bool = True,
        activation = None,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        lora_r: int = 0,
        name="",
    ):
        self.kernel = tf.compat.v1.get_variable(f'{name}_kernel', [in_features, out_features])
        if use_bias:
            self.bias = tf.compat.v1.get_variable(f'{name}_bias', [out_features], initializer=tf.zeros_initializer())
        self.activation = activation

        #print(f'--name: {name}, lora_r: {lora_r}, ')
        if lora_r > 0:
            self.kernel_a = tf.compat.v1.get_variable(f'{name}_lora_kernel_a', [in_features, lora_r], initializer=tf.initializers.he_uniform())
            self.kernel_b = tf.compat.v1.get_variable(f'{name}_lora_kernel_b', [lora_r, out_features], initializer=tf.zeros_initializer())
            self.scaling = lora_alpha / lora_r
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout

        self.use_bias = use_bias
        self.name = name

    def forward(self, x):
        if x.dtype != self.kernel.dtype:
            self.kernel = tf.cast(self.kernel, x.dtype)
            if self.use_bias:
                self.bias = tf.cast(self.bias, x.dtype)
            if self.lora_r > 0:
                self.kernel_a = tf.cast(self.kernel_a, x.dtype)
                self.kernel_b = tf.cast(self.kernel_b, x.dtype)

        kernel = self.kernel
        if self.use_bias:
            bias = self.bias
        if self.lora_r > 0:
            #print(f'[{self.name}] apply lora, self.scaling: {self.scaling}')
            kernel = tf.stop_gradient(kernel)
            kernel += self.kernel_a @ self.kernel_b * self.scaling
            if self.use_bias:
                bias = tf.stop_gradient(self.bias)
        x = x @ kernel
        if self.use_bias:
            x += bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class FeedForward(Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        lora_r: int = 0,
        name="",
    ):
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Dense(dim, hidden_dim, use_bias=False, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_r=lora_r, name=f"{name}_w1")
        self.w2 = Dense(hidden_dim, dim, use_bias=False, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_r=lora_r, name=f"{name}_w2")
        self.w3 = Dense(dim, hidden_dim, use_bias=False, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_r=lora_r, name=f"{name}_w3")

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


@dataclass
class RouterIndices:
    expert_mask: np.ndarray = None
    routing_weights: np.ndarray = None
    #auxiliary_loss
    #router_z_loss


class Router(Module):
    def __init__(self, dim, num_experts, top_k, is_perf_moe_metric, name=""):
        self.num_experts = num_experts
        self.top_k = top_k

        self.route_dense = Dense(in_features=dim, out_features=num_experts, name=name)
        self.is_perf_moe_metric = is_perf_moe_metric

    def forward(self, x):
        """
        x: [bs * seq, dim]
        return: [bs * seq, dim]
        """
        prob = self._compute_router_probabilities(x)
        return self._compute_routing_instructions(prob)

    def _compute_router_probabilities(self, x):
        """
        x: [bs * seq, dim]
        return: [bs * seq, num_experts]
        """
        logit = self.route_dense(x)
        prob = tf.nn.softmax(logit)

        if self.is_perf_moe_metric:
            for idx in range(self.num_experts):
                tf.summary.scalar(f"logit_e{idx}", tf.reduce_mean(logit[:, idx]))
                tf.summary.scalar(f"max_logit_e{idx}", tf.reduce_max(tf.math.abs(logit[:, idx])))
        return prob
    
    def _compute_routing_instructions(self, routing_weights):
        routing_weights, selected_indices = tf.math.top_k(routing_weights, k=self.top_k, sorted=False)
        routing_weights /= tf.reduce_sum(routing_weights, axis=-1, keepdims=True)

        expert_mask = tf.one_hot(selected_indices, depth=self.num_experts)
        expert_mask = tf.transpose(expert_mask, [2, 1, 0]) # [expert, top_k, bs * seq]
        return RouterIndices(expert_mask=expert_mask, routing_weights=routing_weights)


class Expert(Module):
    def __init__(self, expert_id,
                 dim: int,
                 hidden_dim: int,
                 multiple_of: int,
                 ffn_dim_multiplier: Optional[float],
                 name="expert"):
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            name=f'{name}_{expert_id}'
        )

    def forward(self, x):
        """
        x: [bs*seq, dim]
        """
        return self.feed_forward(x)


class MoeLayer(Module):
    def __init__(self, args: ModelArgs, name="moe"):
        self.name = name
        self.num_experts = args.num_experts
        #self.hidden_size = args.hidden_size

        self._router = Router(args.dim, args.num_experts, args.moe_top_k, args.is_perf_moe_metric, name=f'{name}_moe')
        self._experts = [Expert(
            expert_id=i,
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        ) for i in range(args.num_experts)]

        self.is_perf_moe_metric = args.is_perf_moe_metric

    def forward(self, x):
        """
        x: [bs, seq, dim]
        return: [bs, seq, dim]
        """
        bs, seq, dim = shape_list(x)
        num_examples = bs * seq
        x = tf.reshape(x, [num_examples, dim])
        hidden_states = x
        router_indices = self._call_router(hidden_states)
        expert_mask, routing_weights = router_indices.expert_mask, router_indices.routing_weights

        final_hidden_states = tf.zeros([num_examples, dim], dtype=x.dtype)
        for expert_idx in range(len(self._experts)):
            expert_layer = self._experts[expert_idx]
            index = tf.where(expert_mask[expert_idx])  # (c, num_idx), c: n_sample, num_idx: 2
            idx, top_x = index[:, 0], index[:, 1]  # (c)
            current_state = tf.gather(hidden_states, top_x) # (c, dim), c: num_tokens for expert
            current_hidden_states = expert_layer(current_state) # (c, dim)
            #print (current_hidden_states)
            weight_index = tf.stack([top_x, idx], axis=-1)  # (c, 2)
            current_weight = tf.gather_nd(routing_weights, weight_index)  # (c,)
            current_hidden_states *= current_weight[:, None]  # (c, dim)

            final_hidden_states = tf.tensor_scatter_nd_add(final_hidden_states, top_x[:, None], current_hidden_states)       

            if self.is_perf_moe_metric:
                tf.summary.scalar(f"{self.name}_expert_capcity/e_{expert_idx}", tf.reduce_sum(expert_mask[expert_idx]) / tf.cast(bs, tf.float32) / 3 / 6)
        final_hidden_states = tf.reshape(final_hidden_states, [bs, seq, dim])
        return final_hidden_states

    def _call_router(self, x):
        return self._router(x)
