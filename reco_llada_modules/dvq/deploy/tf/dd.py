import tensorflow as tf
import numpy as np
custom_dtype=tf.float32
vocab_size=512
vocab_dim=32
token_num=16

def silu(x):
    return x * tf.sigmoid(x)
tokens = tf.constant([[487, 429, 97, 100, 329, 327, 368, 169, 397, 97, 103, 474, 73, 312, 450, 511]], dtype=tf.int32)

data = np.load("model_weights.npz")
table = tf.constant(data['quantizer.embed.weight'],dtype=custom_dtype)
tokens_one_hot = tf.one_hot(tokens, depth=vocab_size) #[bs, token_num, vocab_size]

z_q = tf.matmul(tokens_one_hot, table) #[bs, token_num, vocab_dim]
mask = tf.expand_dims(tf.cast(tokens != vocab_size, custom_dtype), -1) #[bs, token_num, 1]
z_q_masked = z_q * mask
z_q_masked = tf.reshape(z_q_masked, [-1, token_num * vocab_dim])

# decoder
# print(data['decoder.w1.weight'].shape)
# print(data['decoder.w2.weight'].shape)
# print(data['decoder.w3.weight'].shape)
w1 = tf.constant(data['decoder.w1.weight'].transpose(), dtype=custom_dtype)
w2 = tf.constant(data['decoder.w2.weight'].transpose(), dtype=custom_dtype)
w3 = tf.constant(data['decoder.w3.weight'].transpose(), dtype=custom_dtype)
x1 = tf.matmul(z_q_masked, w1)
x = silu(x1 * tf.matmul(z_q_masked, w3))
out = tf.matmul(x, w2)

with tf.Session() as sess:
    print(sess.run([out]))
