from clsdb_client import clsdb_client
import numpy as np
import struct

client = clsdb_client.ClsdbClient()
succ = client.set_client_param("wxm_mm_sim_gsu_emb", "emb_wxm_mm_sim_gsu_128_new", 0, 128, is_raw_data=True, raw_data_type='uint16') 
# succ = client.set_client_param("recogpt-million-gf17-fp16-sc-hudi-500token", "emb_lyjMillionInterestsCompressScHudi_500token_Ps", 26, 64, is_raw_data=True, raw_data_type='uint16') 

if succ:
  signs = [(10<<48) | 159431160987]
  signs = [159332898962] + [i for i in range(10)]
  x = client.fetch_int_embedding(signs)
  bytes = b''.join(struct.pack('H', value) for value in x)
  x = np.frombuffer(bytes, dtype=np.float16, count=128*len(signs))
  x = np.reshape(x, [-1, 128]).astype(np.float32)
  valid = np.sum(x, axis=1) != 0
  photo_id = np.array(signs)
  photo_id = photo_id[valid]
  # x = client.fetch_float_embedding(signs)
  print("x" * 100)
  print(x[0])
  print(valid)
  print(photo_id)

  # print(signs)
  print("x" * 100)

