# coding=utf-8

import json
import copy

"""
这是一个比较复杂的配置示例。
本示例代码文件为使用 krp 部署 kgnn 集群的示例文件, 尽可能简化配置对于用户来说的复杂度。

通用的 kgnn 集群机器由一组机器组成, 每个机器上存了整个图的一部分(定义为一个 part), 按照 relation-wise 的方式组织。
每个 part 可以存在多个副本以适应不同的训练读写速率, 所有的 part 组合起来为一整个集群.

例如本配置文件展示的异构场景, 集群组成为: [U2I_CLICK, U2I_FOLLOW, I2U_FOLLOW, A2U_CLICK, A2U_LIKE]
按照数据规模切分为: [U2I_CLICK], [U2I_FOLLOW,  I2U_FOLLOW], [A2U_CLICK, A2U_LIKE]
本例子里, 假设了 U2I_CLICK 需要 4 个 shard 存下, 那 U2I_CLICK 占 4 个 part.
另外四种 relation 塞到 2 个 part 里.

krp 部署 kgnn 时的 shard 概念与 part 概念相同, 因此 shard_num = 4+1+1 = 6.
shard0 - shard3 对应 U2I_CLICK 的 4 个 part, shard4 对应存储两种 FOLLOW 的 part, shard5 对应存储两种 A2U 的 part。

这套配置转换脚本可以实现以下操作:
1. 整集群部署, 写好配置后整体部署即可。如果机器数量是集群的 part 数的整数倍, 则每个 part 的副本数都相同.
2. 扩容
- 如果修改 shard_num, 需要修改配置、清除 shm、重新部署。checkpoint 可以处理 shard_num 变化的情况, 服务会自动 reshard。
- 如果只是增加副本，则无需修改配置, 在 krp 上指定 shard 进行扩容即可。
3. 缩容, 在 krp 机器列表指定机器进行下线即可。

可以在本地按照例子执行一下该脚本样例的配置生成, 可以意识到脚本最后实际按照给的 host name 对不同机器做了配置分发,
实际部署的时候 host 列表由 krp 下发到机器上并执行该脚本生成最后服务认识的配置。

在这套脚本内, 每个 part 的最后的 kess name 都不一样, 注意这点, 下面是一个示例, xxx 的部分一定需要用户修改.
"""

###############  用户配置项目  ###################
# 参考：https://git.corp.kuaishou.com/reco-arch/kgnn/-/blob/master/examples/storages/common_cluster_config/cluster_config.py

# 集群名, 用作 kess name 的前缀.
experiment_name = "i2i_bigcode_index"
# 线上单台机器最大可用 shm 内存, 一般来说 c 机器 400G, ae 800G, ae3 1600G
# 比机器的总内存少一些是因为还要留一些给请求缓存等
shm_size = (1 << 30) * 420

# 数组每个 item 代表一个 part 的配置.
part_configs = [
    {
        # 该 part 的 shard 数.
        "shard_num": 4,
        # 该 part 要含有哪些 relation 的数据.
        "dbs": ["I2I"],
        # checkpoint 配置, 图存储定期保存快照
        # "checkpoint": {
        #     # 可以修改为自己的目录配置.
        #     "checkpoint_path": "/home/huxunhan/log/" + experiment_name,
        #     # 最多保留几天的 ckpt 数据.
        #     "reserve_days": 7,
        #     # 保存间隔，默认 6 小时。范围不能超出[3600, 86400]
        #     "save_interval_s": 21600,
        # },
        # 从 hdfs 初始化的配置, 可以配置多个 initer 独立工作, 不需要则删掉该配置.
        # "init": [
        #     {
        #         # 静态初始化配置, 一次性加载数据文件
        #         # 如果静态 Initer 执行前已经有数据（如 shm 未清空, 或从 ckpt 读取了数据）, 则不会执行
        #         # 如果配置多个静态 Initer, 则会依次执行。第二个 Initer 执行时已经有了第一个 Initer 读入的数据, 所以没有实际作用
        #         # 静态 initer 加载完数据之后服务才会启动, 数据量大的情况下推荐使用动态 initer
        #         "type_name": "GnnHdfsIniter",
        #         # 配置路径, 数据库路径, 一般默认就是这个.
        #         "path": "/home/gnn/dw/gnn.db/init_attr_table",
        #         # 分区名, part 1
        #         "service_name": "xxx",
        #         # 分区名, part 2
        #         "relation_name": "I2I",
        #         # （可选）dt 分区读取范围, 闭区间, 默认情况下无限制, 加载目录下所有 dt
        #         "dt_begin": "20220228",
        #         "dt_end": "20220301",
        #         # 并行插入的粒度
        #         "parallel_num": 16,
        #         # 1 代表无权数据格式, 2 代表解析带权图数据
        #         "parse_version": 2,
        #     },
        #     {
        #         # 动态初始化配置, 不断扫描并加载增量文件数据.
        #         "type_name": "GnnHdfsDynamicIniter",
        #         "path": "/home/gnn/dw/gnn.db/init_attr_table",
        #         "service_name": "xxx",
        #         "relation_name": "U2U",
        #         # （可选）dt 分区读取范围, 闭区间, 默认情况下无限制, 加载目录下所有 dt
        #         "dt_begin": "20220228",
        #         "dt_end": "20220228",
        #         # 并行插入的粒度，默认为4。如果只有一个待加载路径，内部会强制使用至多4个线程
        #         # 此限制可通过 --dynamic_parallel_max=4 调整
        #         "parallel_num": 6,
        #         "parse_version": 2,
        #     }
        # ],
        # （可选）从 BTQ 读取数据的配置，不需要可删掉
        # 脚本会自动生成 queue_name 配置，内容是 "btq_kgnn_{exp_name}-{relation_name}-{shard_id}"
        # 目前暂时不支持自动申请 BTQ，需要手动申请
        "stream_loader": {
            "type_name": "BtqStreamLoader",
            "thread_num": 8,
            "pop_num": 10,
        },
    },
    # {
    #     "shard_num": 1,
    #     "dbs": ["U2I_FOLLOW", "I2U_FOLLOW"],
    #     "checkpoint": {
    #         "checkpoint_path": "/home/reco/kgnn/" + experiment_name,
    #         "reserve_days": 5,
    #     }
    # },
    # {
    #     "shard_num": 1,
    #     "dbs": ["A2U_CLICK", "A2U_LIKE"],
    #     "checkpoint": {
    #         "checkpoint_path": "/home/reco/kgnn/" + experiment_name,
    #         "reserve_days": 5,
    #     }
    # }
]

# 针对每个 relation 的配置, key = relation name, value = config.
relations = {
    "I2I": {
        "total_memory": (1 << 30) * 420,
        "key_size": 2000000000,
        "edge_max_num": 100,
        "oversize_replace_strategy": 2,
        "expire_interval": 3600 * 24 * 7,
        "elst": "cpt_weight_indexed_64bit_id",
        # "elst": "cpt_weight_indexed",
        "max_occupancy_rate": 0.9,
        # 权重/出度 衰减相关配置s
        # 边权重衰减比例, 默认为 1(不衰减)
        "weight_decay_ratio": 0.92,
        # 出度累计值的衰减率, 默认为 1(不衰减)
        "degree_decay_ratio": 0.9,
        # 衰减间隔(s), 默认为 86400(1天一次)，evr 的weight是一般是1左右，加权到80。如果是100，在0.9的衰减下，可以在40次左右减少到1，如果没有持续更新，在30小时后就会消失。
        "decay_interval_s": 14400,
        # 边的权重衰减到该值时，淘汰掉此边，默认为 0.
        "delete_threshold_weight": 0.1,
    }
}

###############  用户配置项目  ###################

# 以上仅包含部分常用配置, 更多详细可选配置参考：https://git.corp.kuaishou.com/reco-arch/kgnn/-/wikis/Storage-config/Example

OUTPUT_JSON_CONFIG = "dynamic_json_config.json"


def error_exit(msg):
    # 如果检查不过, 返回的错误信息.
    backup_debug_msg = {"error_msg": msg + ", 请联系 zhaoyonghui, jiangyumeng 咨询"}
    with open(OUTPUT_JSON_CONFIG, "w", encoding="utf8") as f:
        json.dump(backup_debug_msg, f, ensure_ascii=False)
    exit()


# 可调节次级配置.
# 存储的 multi mem kv 的 shard 数, 代表可并行写的并行度.
memkv_shard_num = 64


def translate_db_config(relation_configs):
    for k, v in relation_configs.items():
        v["relation_name"] = k
        # 每个 memkv 的 key 数量
        v["kv_dict_size"] = int(v["key_size"] / memkv_shard_num)
        # 每个 memkv 的内存占用量
        v["kv_size"] = int(v["total_memory"] / memkv_shard_num)
        v["kv_shard_num"] = memkv_shard_num
        v.pop("key_size")
        v.pop("total_memory")
    return relation_configs


def item_name(dbs, shard_id):
    # kess 前缀.
    kess_prefix = "grpc_kgnn_{}".format(experiment_name)
    return "-".join([kess_prefix] + dbs + [str(shard_id)])


def rpc_service_name(dbs):
    # kess 前缀.
    kess_prefix = "grpc_kgnn_{}".format(experiment_name)
    return "-".join([kess_prefix] + dbs)


def update_btq_name(dbs, shard_id):
    # kess 前缀.
    btq_prefix = "btq_kgnn_{}".format(experiment_name)
    return "-".join([btq_prefix] + dbs + [str(shard_id)])


def main():
    with open("host_shard.json", "r") as f:
        host_shard = json.load(f)
    if not host_shard:
        error_exit("host shard load fail")

    if experiment_name is None or experiment_name == "":
        error_exit("experiment_name not set")

    # check 机器数量是否合法.
    krp_shard_num = host_shard["shard_num"]
    total_part_shard_num = sum(
        [
            item["shard_num"] if "shard_num" in item else 1
            for item in part_configs
        ]
    )
    if krp_shard_num != total_part_shard_num:
        error_exit(
            f"not matched shard nums: part shard = {total_part_shard_num}, krp shard = {krp_shard_num}"
        )

    # check 内存分配是否超额度
    for part in part_configs:
        shard = part["shard_num"]
        mem = sum([relations[db]["total_memory"] for db in part["dbs"]])
        if mem / shard > shm_size:
            error_exit(
                f"part with db: {part['dbs']} mem oversize: limit {shm_size}, config: {mem}"
            )

    # key = shard_index, v = host_name
    shard_hosts = {}
    for item in host_shard["hosts"]:
        shard_id = item["shard"]
        if shard_id >= total_part_shard_num:
            error_exit(
                f"shard id exceeds limit = {shard_id}, total shard num = {total_part_shard_num}"
            )
        if shard_id in shard_hosts:
            shard_hosts[shard_id].append(item["host"])
            continue
        shard_hosts[shard_id] = [item["host"]]

    # 分配机器
    host_service_config = {}
    final_config = {}
    final_config["db_list"] = translate_db_config(relations)
    final_config["service_config"] = {"default_rpc_thread_num": 64}

    global_shard_id = -1
    for item in part_configs:
        shard_num = item["shard_num"] if "shard_num" in item else 1
        for shard_id in range(shard_num):
            global_shard_id += 1
            if global_shard_id not in shard_hosts:
                continue
            for host in shard_hosts[global_shard_id]:
                host_service_config[host] = item_name(item["dbs"], shard_id)
                cp_item = copy.deepcopy(item)
                cp_item["service_name"] = rpc_service_name(item["dbs"])
                cp_item["exp_name"] = experiment_name
                cp_item["shard_id"] = shard_id
                cp_item["shard_num"] = shard_num
                # stream loader
                if "stream_loader" in cp_item:
                    cp_item["stream_loader"]["queue_name"] = update_btq_name(
                        item["dbs"], shard_id
                    )
                final_config["service_config"].update(
                    {item_name(item["dbs"], shard_id): cp_item}
                )

    final_config["host_service_config"] = host_service_config

    couple_service = {}
    for k, v in final_config["service_config"].items():
        if type(v) != type({}):
            continue
        for relation in v["dbs"]:
            couple_service[relation] = v["service_name"]

    for k, v in final_config["db_list"].items():
        if "couple_relation_name" in v and "couple_service_name" not in v:
            if v["couple_relation_name"] in couple_service:
                v["couple_service_name"] = couple_service[
                    v["couple_relation_name"]
                ]
            else:
                error_exit(
                    "couple_service_name should be explicitly specified, since we can't find such service in this config file"
                )

    with open(OUTPUT_JSON_CONFIG, "w") as f:
        json.dump(final_config, f, indent=2)


if __name__ == "__main__":
    main()
