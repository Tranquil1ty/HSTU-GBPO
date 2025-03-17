# source ~/activate
# 生成 TopK 采样图
python3 predictTopkSampling.py --cross_att_kv_cache  --self_att_kv_cache --dest_dir predictTopkSampling --fp16
cp predictTopkSampling/* ./server/models/tiger/predictTopkSampling/

# 生成 Beamsearch 图
#python3 predictBeamSearch.py --cross_att_kv_cache  --self_att_kv_cache --dest_dir predictBeamSearch --fp16
#cp predictBeamSearch/* ./server/models/tiger/predictBeamSearch/

# 生成 StepBeamsearch 图
#python3 predictStepBeamSearch.py --cross_att_kv_cache  --self_att_kv_cache --dest_dir predictStepBeamSearch --fp16
#cp predictStepBeamSearch/* ./server/models/tiger/predictStepBeamSearch/
