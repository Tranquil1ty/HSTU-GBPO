#!/usr/bin/env python3
# coding=utf-8
from dragonfly.common_leaf_dsl import LeafFlow

class GenResultFlow(LeafFlow):
    def __init__(self, name):
        LeafFlow.__init__(self, name)

    def _default_flow(self):
        pxtrs = ["evtr", "lvtr", "ltr", "wtr", "wtd", "cmtr", "vtr", "svr", "cpr", "lsst", "epstr"]
        self.if_("muw_enable_use_forumua_one_v2 == 1") \
            .calc_by_formula1(
                kconf_key="formula.scenarioKey15.retrieval_leaf_xtr_filter_f1_nebula",
                export_formula_value = [
                    "f1_score"
                ],
                abtest_biz_name="KUAISHOU_APPS"
            ) \
            .else_() \
            .calc_by_formula1(
                kconf_key="formula.scenarioKey60.retrieval_leaf_xtr_filter_f1_gamora",
                export_formula_value = [
                    "f1_score"
                ],
                abtest_biz_name="KUAISHOU_APPS"
            ) \
            .end_()

        self.perflog_attr_value(
            check_point="{{return 'onerec.pxtr.avg.' .. exp_name}}",
            item_attrs=pxtrs + ["f1_score"],
            aggregator='avg'
        )
        self.perflog_attr_value(
            check_point="{{return 'onerec.pxtr.max.' .. exp_name}}",
            item_attrs=pxtrs + ["f1_score"],
            aggregator='max'
        )
        self.sort_by("f1_score")
        return self
