import tensorflow as tf
import numpy as np

def mult_rotary_emb(self, query_layer, sin, cos):
    # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_query_layer = tf.stack([-query_layer[..., 1::2], query_layer[..., ::2]], axis=-1)
    rotate_half_query_layer = tf.reshape(rotate_half_query_layer, shape_list(query_layer))
    query_layer = query_layer * cos + rotate_half_query_layer * sin
    return query_layer

def rope(q, k, dim, theta=10000):
    """
    q, k: [bs, seq, head, head_dim]
    """
    freqs = tf.convert_to_tensor(1. / (theta ** (np.arange(0, dim, 2)[:(dim // 2)] / dim)), dtype=tf.float32)
    shap = tf.shape(q)
    bs, seq = shap[0], shap[1]
    positions = tf.tile(tf.range(seq)[None, :], [bs, 1])
    positions = positions[:, :, None] # (bs, seq, 1)
    pos_freqs = tf.einsum('..., f -> ... f', tf.cast(positions, dtype=freqs.dtype), freqs) # (bs, seq, 1,) * (..., head_dim//2) -> (bs, seq, 1, head_dim//2)
    pos_freqs = tf.repeat(pos_freqs, 2, axis=-1)
    sin, cos = tf.sin(pos_freqs), tf.cos(pos_freqs)
    q2 = mult_rotary_emb(q, sin, cos)
    k2 = mult_rotary_emb(k, sin, cos)
    return q2, k2

def mask(inputs, key_masks=None, type=None):
    padding_num = -2 ** 32 + 1
    if type in ("key"):
        key_masks = tf.to_float(key_masks) # (N, seqlen)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
        print("--------key-----------")
    if type in ("future"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)
        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        print("-------future-----------")

    return outputs

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * tf.pow(x,3))))

def dropout(inputs, dropout_rate, training):
    return tf.layers.dropout(inputs, rate=dropout_rate, training=training)

# t5 style: no bias and no subtraction of mean
def layernorm(hidden_states,
              epsilon=1e-6,
              reuse=None):
    with tf.variable_scope("layer_norm", reuse=reuse):
        w = tf.get_variable('w', (hidden_states.get_shape()[-1],), initializer=tf.constant_initializer(1.0))
        variance = tf.math.reduce_mean(tf.math.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * tf.math.rsqrt(variance + epsilon)
        return w * hidden_states

def rms_norm(x, eps=1e-8, p=-1., bias=False, scope=None):
    with tf.variable_scope(scope or "rms_norm"):
        layer_size = x.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        if bias:
            offset = tf.get_variable("offset", [layer_size], initializer=tf.zeros_initializer())
        else:
            offset = 0.

        if p < 0. or p > 1.:
            ms = tf.reduce_mean(x ** 2, -1, keep_dims=True)
        else:
            partial_size = int(layer_size * p)
            partial_x, _ = tf.split(x, [partial_size, layer_size - partial_size], axis=-1)

            ms = tf.reduce_mean(partial_x ** 2, -1, keep_dims=True)

        return scale * x * tf.rsqrt(ms + eps) + offset

def ffn(hidden_states,
        d_model,
        d_ff,
        dropout_rate=0.1,
        training=True,
        reuse=None):
    with tf.variable_scope("FFN", reuse=reuse):
        wi_initializer = tf.random_normal_initializer(mean=0, stddev=d_model**-0.5)
        wo_initializer = tf.random_normal_initializer(mean=0, stddev=d_ff**-0.5)

        wi = tf.get_variable('wi', (d_model, d_ff), initializer=wi_initializer)
        wo = tf.get_variable('wo', (d_ff, d_model), initializer=wo_initializer)

        # pre norm
        normed_hidden_states = rms_norm(hidden_states)

        dense_output = tf.tensordot(normed_hidden_states, wi, axes=(-1, 0))
        dense_output = gelu(dense_output)
        dense_output = dropout(dense_output, dropout_rate, training)

        dense_output = tf.tensordot(dense_output, wo, axes=(-1, 0))
        dense_output = dropout(dense_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + dense_output

        return hidden_states

def self_attention(hidden_states,
                   d_model,
                   d_kv,
                   n_heads,
                   dropout_rate=0.1,
                   key_masks=None,
                   causality=False,
                   training=True,
                   reuse=None):
    with tf.variable_scope("self_attention", reuse=reuse):
        inner_dim = d_kv * n_heads

        q_initializer = tf.random_normal_initializer(mean=0, stddev=(inner_dim * d_kv)**-0.5)
        k_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        v_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        o_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)

        q = tf.get_variable('q', (d_model, inner_dim), initializer=q_initializer)
        k = tf.get_variable('k', (d_model, inner_dim), initializer=k_initializer)
        v = tf.get_variable('v', (d_model, inner_dim), initializer=v_initializer)
        o = tf.get_variable('o', (inner_dim, d_model), initializer=o_initializer)

         # pre_norm
        normed_hidden_states = rms_norm(hidden_states)

        query_states = tf.tensordot(normed_hidden_states, q, axes=(-1, 0))
        key_states = tf.tensordot(normed_hidden_states, k, axes=(-1, 0))
        value_states = tf.tensordot(normed_hidden_states, v, axes=(-1, 0))

        query_states = tf.concat(tf.split(query_states, n_heads, axis=2), axis=0)
        key_states = tf.concat(tf.split(key_states, n_heads, axis=2), axis=0)
        value_states = tf.concat(tf.split(value_states, n_heads, axis=2), axis=0)

        scores = tf.matmul(query_states / (d_kv ** 0.5), tf.transpose(key_states, [0, 2, 1])) # (h*N, T_q, T_k)

        # key_mask
        if key_masks is not None:
            scores = mask(scores, key_masks, type="key")

        if causality:
            scores = mask(scores, type="future")

        weights = tf.nn.softmax(scores)
        weights = dropout(weights, dropout_rate, training)

        attn_output = tf.matmul(weights, value_states)
        attn_output = tf.concat(tf.split(attn_output, n_heads, axis=0), axis=2)
        attn_output = tf.tensordot(attn_output, o, axes=(-1, 0))
        attn_output = dropout(attn_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + attn_output

        return hidden_states

def cross_attention(hidden_states,
                    key_value_states,
                    d_model,
                    d_kv,
                    n_heads,
                    dropout_rate=0.1,
                    key_masks=None,
                    causality=False,
                    training=True,
                    reuse=None):
    with tf.variable_scope("cross_attention", reuse=reuse):
        inner_dim = d_kv * n_heads

        q_initializer = tf.random_normal_initializer(mean=0, stddev=(inner_dim * d_kv)**-0.5)
        k_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        v_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        o_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)

        q = tf.get_variable('q', (d_model, inner_dim), initializer=q_initializer)
        k = tf.get_variable('k', (d_model, inner_dim), initializer=k_initializer)
        v = tf.get_variable('v', (d_model, inner_dim), initializer=v_initializer)
        o = tf.get_variable('o', (inner_dim, d_model), initializer=o_initializer)

         # pre_norm
        normed_hidden_states = rms_norm(hidden_states)

        query_states = tf.tensordot(normed_hidden_states, q, axes=(-1, 0))
        key_states = tf.tensordot(key_value_states, k, axes=(-1, 0))
        value_states = tf.tensordot(key_value_states, v, axes=(-1, 0))

        query_states = tf.concat(tf.split(query_states, n_heads, axis=2), axis=0)
        key_states = tf.concat(tf.split(key_states, n_heads, axis=2), axis=0)
        value_states = tf.concat(tf.split(value_states, n_heads, axis=2), axis=0)

        scores = tf.matmul(query_states / (d_kv ** 0.5), tf.transpose(key_states, [0, 2, 1])) # (h*N, T_q, T_k)

        # key_mask
        if key_masks is not None:
            scores = mask(scores, key_masks, type="key")

        if causality:
            scores = mask(scores, type="future")

        weights = tf.nn.softmax(scores)
        weights = dropout(weights, dropout_rate, training)

        attn_output = tf.matmul(weights, value_states)
        attn_output = tf.concat(tf.split(attn_output, n_heads, axis=0), axis=2)
        attn_output = tf.tensordot(attn_output, o, axes=(-1, 0))
        attn_output = dropout(attn_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + attn_output

        return hidden_states

def flash_attention(hidden_states,
                    d_model,
                    d_kv,
                    n_heads,
                    config,
                    dropout_rate=0.1,
                    key_masks=None,
                    causality=False,
                    training=True,
                    reuse=None):
    with tf.variable_scope("self_attention", reuse=reuse):
        inner_dim = d_kv * n_heads

        q_initializer = tf.random_normal_initializer(mean=0, stddev=(inner_dim * d_kv)**-0.5)
        k_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        v_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        o_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)

        q = tf.get_variable('q', (d_model, inner_dim), initializer=q_initializer)
        k = tf.get_variable('k', (d_model, inner_dim), initializer=k_initializer)
        v = tf.get_variable('v', (d_model, inner_dim), initializer=v_initializer)
        o = tf.get_variable('o', (inner_dim, d_model), initializer=o_initializer)

         # pre_norm
        normed_hidden_states = rms_norm(hidden_states)

        # (batch_size, seq, head*dim)
        query_states = tf.tensordot(normed_hidden_states, q, axes=(-1, 0))
        key_states = tf.tensordot(normed_hidden_states, k, axes=(-1, 0))
        value_states = tf.tensordot(normed_hidden_states, v, axes=(-1, 0))

        # query_states = query_states / (d_kv ** 0.5)

        # no mask
        # no dropout

        seqinfo = tf.fill([tf.shape(query_states)[0]], query_states.get_shape().as_list()[1]) # int32

        #seqinfo = tf.Print(seqinfo, [seqinfo, tf.shape(seqinfo)], "=====> seqinfo:")
        #query_states = tf.Print(query_states, [query_states, tf.shape(query_states), tf.shape(tf.reshape(query_states, (1, -1, n_heads, d_kv)))], "=====> query_states:")
        #key_states = tf.Print(key_states, [key_states, tf.shape(key_states), tf.shape(tf.reshape(key_states, (1, -1, n_heads, d_kv)))], "=====> key_states:")
        #value_states = tf.Print(value_states, [value_states, tf.shape(value_states), tf.shape(tf.reshape(value_states, (1, -1, n_heads, d_kv)))], "=====> value_states:")

        attn_output = config.mem_eff_attn(
            tf.reshape(query_states, (1, -1, n_heads, d_kv)),
            tf.reshape(key_states, (1, -1, n_heads, d_kv)),
            tf.reshape(value_states, (1, -1, n_heads, d_kv)),
            bf16=True,
            q_seqinfo=seqinfo,
            k_seqinfo=seqinfo,
            custom_mask_type=config.nn.CustomMaskType.BlockDiagonalMask,
            use_v2=True
        )

        attn_output = tf.reshape(attn_output, (-1, query_states.get_shape().as_list()[1], inner_dim))
        attn_output = tf.tensordot(attn_output, o, axes=(-1, 0))
        attn_output = dropout(attn_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + attn_output

        return hidden_states
    
def cross_flash_attention(hidden_states,
                    key_value_states,
                    d_model,
                    d_kv,
                    n_heads,
                    config,
                    dropout_rate=0.1,
                    key_masks=None,
                    causality=False,
                    training=True,
                    reuse=None):
    with tf.variable_scope("cross_attention", reuse=reuse):
        inner_dim = d_kv * n_heads

        q_initializer = tf.random_normal_initializer(mean=0, stddev=(inner_dim * d_kv)**-0.5)
        k_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        v_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)
        o_initializer = tf.random_normal_initializer(mean=0, stddev=inner_dim**-0.5)

        q = tf.get_variable('q', (d_model, inner_dim), initializer=q_initializer)
        k = tf.get_variable('k', (d_model, inner_dim), initializer=k_initializer)
        v = tf.get_variable('v', (d_model, inner_dim), initializer=v_initializer)
        o = tf.get_variable('o', (inner_dim, d_model), initializer=o_initializer)

         # pre_norm
        normed_hidden_states = rms_norm(hidden_states)

        # (batch_size, seq, head*dim)
        query_states = tf.tensordot(normed_hidden_states, q, axes=(-1, 0))
        key_states = tf.tensordot(key_value_states, k, axes=(-1, 0))
        value_states = tf.tensordot(key_value_states, v, axes=(-1, 0))

        # query_states = query_states / (d_kv ** 0.5)

        # no mask
        # no dropout

        q_seqinfo = tf.fill([tf.shape(query_states)[0]], query_states.get_shape().as_list()[1]) # int32
        k_seqinfo = tf.fill([tf.shape(key_value_states)[0]], key_value_states.get_shape().as_list()[1]) # int32

        #seqinfo = tf.Print(seqinfo, [seqinfo, tf.shape(seqinfo)], "=====> seqinfo:")
        #query_states = tf.Print(query_states, [query_states, tf.shape(query_states), tf.shape(tf.reshape(query_states, (1, -1, n_heads, d_kv)))], "=====> query_states:")
        #key_states = tf.Print(key_states, [key_states, tf.shape(key_states), tf.shape(tf.reshape(key_states, (1, -1, n_heads, d_kv)))], "=====> key_states:")
        #value_states = tf.Print(value_states, [value_states, tf.shape(value_states), tf.shape(tf.reshape(value_states, (1, -1, n_heads, d_kv)))], "=====> value_states:")

        attn_output = config.mem_eff_attn(
            tf.reshape(query_states, (1, -1, n_heads, d_kv)),
            tf.reshape(key_states, (1, -1, n_heads, d_kv)),
            tf.reshape(value_states, (1, -1, n_heads, d_kv)),
            bf16=True,
            q_seqinfo=q_seqinfo,
            k_seqinfo=k_seqinfo,
            custom_mask_type=config.nn.CustomMaskType.BlockDiagonalMask,
            use_v2=True
        )

        attn_output = tf.reshape(attn_output, (-1, query_states.get_shape().as_list()[1], inner_dim))
        attn_output = tf.tensordot(attn_output, o, axes=(-1, 0))
        attn_output = dropout(attn_output, dropout_rate, training)

        # residual
        hidden_states = hidden_states + attn_output

        return hidden_states