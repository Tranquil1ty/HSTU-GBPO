from dragonfly.matx.dragonfly_context import DragonflyContext
import random
import math
from typing import List as FTList
from typing import Dict as FTDict

class RAGFuncSet:
    def __init__(self) -> None:
        self.shift_num: int = 30000

    def sign_to_slot(self, ctx: DragonflyContext) -> None:
        shift_num = ctx.GetInt(b"current_step", self.shift_num)

        item_rag_parameters_getter = ctx.ItemAttrGetter(b"item_rag_parameters")
        item_rag_slots_setter = ctx.ItemAttrSetter(b"item_rag_slots")

        result_size = ctx.GetItemNum()
        for idx in range(result_size):
            item_rag_parameters = item_rag_parameters_getter.GetIntList(idx)

            item_rag_slots: FTList[int] = []

            for i in range(len(item_rag_parameters)):
                slot = (item_rag_parameters[i] >> 48) + 16 + shift_num
                item_rag_slots.append(slot)
        
            item_rag_slots_setter.SetIntList(idx, item_rag_slots)
