# 多模态 Item Alignment Emb I2I 召回

## 代码结构
- [ann](./ann/): ANN 近邻 TopK 检索 -> [线上服务](https://kaiserving.corp.kuaishou.com/deployment-model/own-model/information?modelId=52019)

- [embserver](./embserver/): 存储 pid 对应的多模态 emb -> [线上服务](https://krp.corp.kuaishou.com/part/43363)

- [kgnn](./kgnn/): 缓存 <pid, sim_pids> 的 kv 对，sim_pids 是 ANN 检索的相似结果 -> 本服务没有部署，参考 [线上服务](https://krp.corp.kuaishou.com/part/58297)

- [runner](./runner/): 遍历推荐侧有效的视频索引池，【请求 ANN 写入 kgnn，本代码中没有启用】-> 部署服务[线上服务](https://krp.corp.kuaishou.com/part/63277)

- [retr_flow](./retr_flow/): 接收用户请求，召回服务返回 i2i 召回结果 -> 部署服务 [线上服务](https://krp.corp.kuaishou.com/part/63214)


## 整体调用关系
[Item Alignment 召回](https://docs.corp.kuaishou.com/k/home/VGpWrKyU_38w/fcADnauwBz3E4PgV7hNf2dGhs?ro=false)
![](https://docs.corp.kuaishou.com/image/api/external/load/out?code=fcADnauwBz3E4PgV7hNf2dGhs:-923756690652567664fcADnauwBz3E4PgV7hNf2dGhs:1726301566698)

注：本图中没有展示 runner 写 KGNN 的流程。


## 其他

带 KGNN 写入的架构，参考 [粗排双塔I2I](https://docs.corp.kuaishou.com/d/home/fcABeUcrUlc54uZQ86OYhckIC)
<font color="#000000">![](https://docs.corp.kuaishou.com/image/api/external/load/out?code=fcABeUcrUlc54uZQ86OYhckIC:3952217051522431918fcABeUcrUlc54uZQ86OYhckIC:1726301787826)</font> 

对应代码位置：https://git.corp.kuaishou.com/retrieval-models/projects/-/tree/master/i2i_center/i2i_index/mc_tower?ref_type=heads