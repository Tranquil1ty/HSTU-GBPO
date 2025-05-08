from dragonfly.matx.dragonfly_context import DragonflyContext
import random
import math
from typing import List as FTList
from typing import Dict as FTDict

class ChooseTokenStrategy:
    def __init__(self) -> None:
        self.token_num: int = 7
        self.vocab_size: int = 64
        self.default_topk: int = 20
        self.default_topp: float = 0.9
        self.default_temp: float = 1.0
    
    def _sort_by_prob(self, x: dict) -> float:
        return x["prob"]

    def softmax(self, logits: list, temp: float) -> list:
        probs = []
        sum_exp = 0.0
        sum_exp_ori = 0.0
        for i in range(len(logits)):
            sum_exp += math.exp(logits[i] / temp)
            sum_exp_ori += math.exp(logits[i])

        for i in range(len(logits)):
            probs.append({
                "index": i,
                "prob": math.exp(logits[i] / temp) / sum_exp,
                "ori_prob": math.exp(logits[i]) / sum_exp_ori
            })
        return probs
    
    def prob_sample(self, probs_i: list) -> dict:
        probs_sum = 0.0
        rand_val = random.random()
        for i in range(len(probs_i)):
            probs_sum += (probs_i[i]["prob"])
            if rand_val < probs_sum:
                return probs_i[i]
        return probs_i[-1]
    
    def prob_sample_index(self, probs: FTList[float]) -> int:
        probs_sum = 0.0
        rand_val = random.random()
        for i in range(len(probs)):
            probs_sum += (probs[i])
            if rand_val <= probs_sum:
                return i
        return len(probs) - 1
    
    def topk_sample(self, probs_i: list, topk: int) -> list:
        probs_i.sort(key=self._sort_by_prob, reverse=True)
        return probs_i[:topk]
    
    def topp_sample(self, probs_i: list, topp: float) -> list:
        result = []
        probs_sum = 0.0
        for i in range(len(probs_i)):
            if probs_i[i]["prob"] + probs_sum < topp:
                probs_sum += probs_i[i]["prob"]
                result.append(probs_i[i])
            else:
                break
        if len(result) == 0:
            result.append(probs_i[0])
        return result
    
    def topp_sample_index(self, probs: FTList[float], topp: float) -> int:
        probs_sum = 0.0
        for i in range(len(probs)):
            if probs[i] + probs_sum < topp:
                probs_sum += probs[i]
            else:
                return i
        return len(probs) - 1
    
    def prob_norm(self, probs_i: list) -> list:
        probs_sum = 0.0
        for i in range(len(probs_i)):
            probs_sum += (probs_i[i]["prob"])

        for i in range(len(probs_i)):
            probs_i[i]["prob"] = probs_i[i]["prob"] / probs_sum
        return probs_i
    
    def prob_norm_v2(self, probs: FTList[float]) -> FTList[float]:
        probs_sum = 0.0
        for i in range(len(probs)):
            probs_sum += (probs[i])

        for i in range(len(probs)):
            probs[i] = probs[i] / probs_sum
        return probs
    
    def choose_token_beam(self, ctx: DragonflyContext) -> None:
        current_step = ctx.GetInt(b"current_step")

        topk_prob_getter = ctx.ItemAttrGetter(b"topk_prob")
        topk_indices_getter = ctx.ItemAttrGetter(b"topk_indices")
        semantic_id_v2_getter = ctx.ItemAttrGetter(b"semantic_id_v2")
        semantic_id_v2_prob_getter = ctx.ItemAttrGetter(b"semantic_id_v2_prob")
        beam_prob_getter = ctx.ItemAttrGetter(b"beam_prob")

        semantic_id_v2_setter = ctx.ItemAttrSetter(b"semantic_id_v2")
        semantic_id_v2_prob_setter = ctx.ItemAttrSetter(b"semantic_id_v2_prob")
        beam_prob_setter = ctx.ItemAttrSetter(b"beam_prob")
       
        
        pre_seq_indices: FTList[int] = []
        token_indices: FTList[int] = []
        token_probs: FTList[float] = []
        pre_seqs: FTList[FTList[float]] = []
        pre_seq_probs: FTList[FTList[float]] = []
        beam_probs: FTList[float] = []

        result_size = ctx.GetItemNum()
        for i in range(result_size):
            prob = topk_prob_getter.GetDouble(i)
            idx = topk_indices_getter.GetInt(i)

            pre_seq_input = semantic_id_v2_getter.GetDoubleList(i)
            pre_seq_input_prob = semantic_id_v2_prob_getter.GetDoubleList(i)

            tmp_pre_seq: FTList[float] = []
            tmp_pre_seq_prob: FTList[float] = []
            for j in range(self.token_num):
                tmp_pre_seq.append(pre_seq_input[j])
                tmp_pre_seq_prob.append(pre_seq_input_prob[j])
            pre_seqs.append(tmp_pre_seq)
            pre_seq_probs.append(tmp_pre_seq_prob)

            token_indices.append(idx % self.vocab_size)
            token_probs.append(prob)
            pre_seq_indices.append(idx // self.vocab_size)
            beam_probs.append(beam_prob_getter.GetDoubleList(i)[0])
        
        for i in range(result_size):
            choosed_token_probs: FTList[float] = []
            choosed_token_ids: FTList[float] = []

            token_idx = token_indices[i]
            token_prob = token_probs[i]

            pre_seq_idx = pre_seq_indices[i]
            pre_seq = pre_seqs[pre_seq_idx]
            pre_seq_prob = pre_seq_probs[pre_seq_idx]

            beam_prob = beam_probs[pre_seq_idx]
            beam_prob_new: FTList[float] = []
            beam_prob_new.append(beam_prob + math.log(token_prob))

            for j in range(self.token_num):
                if j == current_step:
                    choosed_token_probs.append(token_prob * 1.0)
                    choosed_token_ids.append(token_idx * 1.0)
                else:
                    choosed_token_probs.append(pre_seq_prob[j])
                    choosed_token_ids.append(pre_seq[j])

            semantic_id_v2_setter.SetDoubleList(i, choosed_token_ids)
            semantic_id_v2_prob_setter.SetDoubleList(i, choosed_token_probs)
            beam_prob_setter.SetDoubleList(i, beam_prob_new)

    def choose_token_v2(self, ctx: DragonflyContext) -> None:
        p_topk = ctx.GetInt(b"p_topk", self.default_topk)
        p_topp = ctx.GetDouble(b"p_topp", self.default_topp)
        p_temp = ctx.GetDouble(b"p_temp", self.default_temp)

        pre_topk = 50

        topk_prob_getter = ctx.ItemAttrGetter(b"topk_prob")
        topk_indices_getter = ctx.ItemAttrGetter(b"topk_indices")

        token_probs_setter = ctx.ItemAttrSetter(b"token_probs")
        token_ids_setter = ctx.ItemAttrSetter(b"token_indices")

        result_size = ctx.GetItemNum()
        for idx in range(result_size):
            pre_topk_prob = topk_prob_getter.GetDoubleList(idx)
            pre_topk_indices = topk_indices_getter.GetIntList(idx)

            choosed_token_probs: FTList[float] = []
            choosed_token_ids: FTList[int] = []

            for i in range(self.token_num):
                c_prob = pre_topk_prob[i * pre_topk:(i + 1) * pre_topk]
                c_indices = pre_topk_indices[i * pre_topk:(i + 1) * pre_topk]

                # tensorflow topk 已经排序了
                topk_prob = c_prob[:p_topk]
                topk_indices = c_indices[:p_topk]

                topk_prob = self.prob_norm_v2(topk_prob)

                topp_index = self.topp_sample_index(topk_prob, p_topp)

                topp_prob = topk_prob[:topp_index+1]
                topp_indices = topk_indices[:topp_index+1]

                topp_prob = self.prob_norm_v2(topp_prob)

                sidx = self.prob_sample_index(topp_prob)

                choosed_token_ids.append(topp_indices[sidx])
                choosed_token_probs.append(topp_prob[sidx])
        
            token_probs_setter.SetDoubleList(idx, choosed_token_probs)
            token_ids_setter.SetIntList(idx, choosed_token_ids)

    def choose_token(self, ctx: DragonflyContext) -> None:
        p_topk = ctx.GetInt(b"p_topk", self.default_topk)
        p_topp = ctx.GetDouble(b"p_topp", self.default_topp)
        p_temp = ctx.GetDouble(b"p_temp", self.default_temp)

        logits_getter = ctx.ItemAttrGetter(b"logits")

        token_probs_setter = ctx.ItemAttrSetter(b"token_probs")
        token_ids_setter = ctx.ItemAttrSetter(b"token_indices")

        result_size = ctx.GetItemNum()
        for idx in range(result_size):
            logits = logits_getter.GetDoubleList(idx)

            choosed_token_probs: FTList[float] = []
            choosed_token_ids: FTList[int] = []

            for i in range(self.token_num):
                logits_i = []
                for j in range(self.vocab_size):
                    logits_i.append(logits[i * self.vocab_size + j]) 

                probs_i = self.softmax(logits_i, p_temp)

                # topk sample
                if p_topk > 0 and p_topk < self.vocab_size:
                    probs_i = self.topk_sample(probs_i, p_topk)
                    probs_i = self.prob_norm(probs_i)
                
                # topp sample
                if p_topp > 0 and p_topp < 1.0:
                    probs_i = self.topp_sample(probs_i, p_topp)
                    probs_i = self.prob_norm(probs_i)
                
                res = self.prob_sample(probs_i)
                choosed_token_probs.append(res["ori_prob"])
                choosed_token_ids.append(res["index"])
            
            token_probs_setter.SetDoubleList(idx, choosed_token_probs)
            token_ids_setter.SetIntList(idx, choosed_token_ids)

class RemaskTokenStrategy:
    def __init__(self) -> None:
        self.token_num: int = 7
        self.vocab_size: int = 64
        self.mask_token_id: int = self.vocab_size
        self.default_infer_step: int = 8
    
    def _get_prob_for_sorted(self, x: dict) -> float:
        return x["pred_token_prob"]
    
    def remask_token(self, ctx: DragonflyContext) -> None:
        infer_step = ctx.GetInt(b"infer_step", self.default_infer_step)
        current_step = ctx.GetInt(b"current_step", 0)
        random_remask_flag = ctx.GetInt(b"random_remask_flag", 0)

        t = 1.0 - (current_step * 1.0) / infer_step
        s = t - (1.0 / infer_step)
        mask_token_num = int(self.token_num * s)
        mask_ratio = s / t

        semantic_id_v2_getter = ctx.ItemAttrGetter(b"semantic_id_v2")
        semantic_id_v2_mask_getter = ctx.ItemAttrGetter(b"semantic_id_v2_mask")

        token_probs_getter = ctx.ItemAttrGetter(b"token_probs")
        token_ids_getter = ctx.ItemAttrGetter(b"token_indices")

        semantic_id_v2_setter = ctx.ItemAttrSetter(b"semantic_id_v2")
        semantic_id_v2_mask_setter = ctx.ItemAttrSetter(b"semantic_id_v2_mask")

        result_size = ctx.GetItemNum()
        for idx in range(result_size):
            semantic_id_v2 = semantic_id_v2_getter.GetDoubleList(idx)
            semantic_id_v2_mask = semantic_id_v2_mask_getter.GetDoubleList(idx)
            token_probs = token_probs_getter.GetDoubleList(idx)
            token_ids = token_ids_getter.GetIntList(idx)

            result_token_ids: FTList[float] = []
            result_token_mask: FTList[float] = []

            mask_token_info = []
            for token_pos, input_token_id in enumerate(semantic_id_v2):
                result_token_ids.append(semantic_id_v2[token_pos])
                result_token_mask.append(semantic_id_v2_mask[token_pos])
                if semantic_id_v2_mask[token_pos] == 1.0:
                    mask_token_info.append({
                        "idx": token_pos,
                        "input_token_id": input_token_id,
                        "pred_token_id": token_ids[token_pos],
                        "pred_token_prob": token_probs[token_pos]
                    })
                    result_token_ids[token_pos] = token_ids[token_pos] * 1.0
                    result_token_mask[token_pos] = 0.0
            

            remask_token_info = []
            # random remask
            if random_remask_flag == 1:
                for mti in mask_token_info:
                    if random.random() < mask_ratio:
                        remask_token_info.append(mti)
            # low confidence remask
            else:
                sorted_info = sorted(mask_token_info, key=self._get_prob_for_sorted, reverse=False)
                for i in range(mask_token_num):
                    if i < len(sorted_info):
                        remask_token_info.append(sorted_info[i])
            
            for item in remask_token_info:
                pos_idx = item["idx"]
                result_token_ids[pos_idx] = self.mask_token_id * 1.0
                result_token_mask[pos_idx] = 1.0
            
            semantic_id_v2_setter.SetDoubleList(idx, result_token_ids)
            semantic_id_v2_mask_setter.SetDoubleList(idx, result_token_mask)
