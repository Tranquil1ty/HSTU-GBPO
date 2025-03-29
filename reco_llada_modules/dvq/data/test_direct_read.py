import unittest
from unittest import mock
import argparse
from direct_read_data import EmbeddingData, UserSeqDataset, ParquetData
from kafka_fetch_data_clsdb_client import MissedKeyData

# class TestEmbeddingData(unittest.TestCase):

#     def test_train_dataloader(self):
#         args = argparse.Namespace(
#             data_dir='/llm_reco_ssd/luoxinchen/dinov2/features/ia_emb_20241117',
#             batch_size=128,
#             num_workers=8)
#         data = EmbeddingData(args.data_dir, args.batch_size, args.num_workers)
#         dl = data.train_dataloader()
#         for s in dl:
#             print(s)

# class TestUserSeqDataset(unittest.TestCase):

#     def test_iter(self):
#         ds = UserSeqDataset(
#             '/llm_reco_ssd/luoxinchen/dinov2/features/ia_emb_short1k_20241206_merged/',
#             '/llm_reco_ssd/luoxinchen/end2endreco/data/short1k_hist/20241206/2024120620.txt',
#         )
#         for s in ds:
#             print(s)


# class TestParquetDataset(unittest.TestCase):

#     def test_iter(self):
#         data = ParquetData("viewfs://hadoop-lt-cluster/home/reco_kaiworks/dw/reco_kaiworks.db/rlj_semantic_id_training_samples/p_date=20250226", 1024, 2)
#         dl = data.train_dataloader()
#         for s in dl:
#             print(s[0].shape, s[1].shape)
#             break

class TestKafkaDataset(unittest.TestCase):

    def test_iter(self):

        clsdb_cfg = dict(model='wxm_mm_sim_gsu_emb', table='emb_wxm_mm_sim_gsu_128_new', dtype='float16', dim=128)
        data_module = MissedKeyData('hierarchical_semantic_id_v1_missed_key', "rlj_debug_consume", clsdb_cfg, 60, None, num_workers=0)
        dl = data_module.test_dataloader()
        for s in dl:
            _, _, all_num, valid_num, keys = s

            print(f"{valid_num} / {all_num} [{len(keys)}] [{len(list(set(keys)))}]")
            input()

if __name__ == '__main__':
    unittest.main()