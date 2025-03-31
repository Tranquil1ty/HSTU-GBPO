from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.common.common_api_mixin import CommonApiMixin
from dragonfly.common_leaf_dsl import LeafFlow

from dragonfly.matx.dragonfly_context import DragonflyContext

class EvalFunctions:

    def __init__(self) -> None:
        pass

    def calc_hit_rate(self, ctx: DragonflyContext) -> None:
        result_pids = ctx.GetIntList(b"result_pids")
        pos_pids = ctx.GetIntList(b"eval_pos_photo_id_list")
        is_hit = False
        pos_pids_set = set()
        for pid in pos_pids:
            pos_pids_set.add(pid)
        for pid in result_pids:
            if pid in pos_pids_set:
                is_hit = True
                break
        ctx.SetInt(b"is_hit", is_hit)


class HitRatePerfFlow(LeafFlow,OfflineApiMixin, CommonApiMixin):

    def hit_rate_perf(self):
        self.if_("eval_pos_photo_id_list ~= nil")
        self.pack_item_attr(
            item_source={
                "reco_results": True,
            },
            mappings=[
                {
                    "from_item_attr": "pid",
                    "to_common_attr": "result_pids",
                }
            ]
        )
        self._calc_hit_rate()
        self.log_debug_info(
            common_attrs=["is_hit", "eval_pos_photo_id_list", "result_pids"],
            for_debug_request_only=False,
        )
        self.copy_attr(
            attrs=[
                {"from_common": "is_hit", "to_common": "hit_rate"},
            ]
        )
        self.perflog_attr_value(check_point="generative.hit_rate", common_attrs=["hit_rate"])
        self.end_if_()
        return self
    
    def _calc_hit_rate(self):
        self.enrich_attr_by_py(
            function_set=EvalFunctions,
            py_function=EvalFunctions.calc_hit_rate,
        )
        return self