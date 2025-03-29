from kess.framework import (
    ClientOption,
    GrpcClient,
    KessOption
)
from proto.embedding_server_pb2 import GetKuibaEmbeddingRequest
from proto.embedding_server_pb2_grpc import *

class EmbClient(object):
    def __init__(self, 
                 biz_df, 
                 grpc_service_name, 
                 shards,
                 servicer_name="ai-platform-dense-server"):
        kess_option = KessOption(biz_def=biz_df, name=servicer_name,port='0000')
        self.client_option=ClientOption(
            biz_def=biz_df,
            grpc_service_name=grpc_service_name,
            grpc_stub_class=PredictKessServiceStub,
            servicer_option=kess_option,
        )
        client = GrpcClient(self.client_option)
        self.clients = []
        for i in range(shards):
            self.clients.append(client.select_shard("s%d"%(i)))
        self.shards = shards
    
    def _do_get_emb(self, pids):
        pid_emb = dict()
        responses = []
        for i in range(self.shards):
            req = GetKuibaEmbeddingRequest()
            for pid in pids:
                idx = int(pid) % self.shards
                if idx == i:
                    req.signs.append(int(pid))
            res = self.clients[i].GetKuibaEmbedding.future(req, timeout=60.0)
            responses.append(res)
        for res in responses:
            res = res.result()
            for sign, value in zip(res.signs, res.values):
                pid_emb[sign] = value
        return pid_emb

    def get_emb(self, pids):
        r = dict()
        for i in range(3):
            try:
                r = self._do_get_emb(pids)
                return r
            except Exception as e:
                print(e)
                continue
        return r
