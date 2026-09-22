#!/usr/bin/env python3

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mixed"))

from splice_mixed_expert_layers_gguf import should_take_donor, tensor_nbytes


class SpliceSelectionTest(unittest.TestCase):
    def test_default_selects_routed_experts_in_layers(self):
        self.assertTrue(should_take_donor("blk.17.ffn_down_exps.weight", {17}))
        self.assertFalse(should_take_donor("blk.16.ffn_down_exps.weight", {17}))
        self.assertFalse(should_take_donor("blk.17.kda_v.weight", {17}))

    def test_custom_regex_selects_kda_projections_only(self):
        pattern = re.compile(r"^blk\.(\d+)\.kda_(v|output)\.weight$")
        self.assertTrue(should_take_donor("blk.0.kda_v.weight", {0}, pattern))
        self.assertTrue(should_take_donor("blk.0.kda_output.weight", {0}, pattern))
        self.assertFalse(should_take_donor("blk.0.kda_v_conv.weight", {0}, pattern))
        self.assertFalse(should_take_donor("blk.0.kda_q.weight", {0}, pattern))
        self.assertFalse(should_take_donor("blk.17.ffn_down_exps.weight", {17}, pattern))

    def test_glm_dense_types_have_sizes(self):
        self.assertEqual(tensor_nbytes((4096, 64), 30), 4096 * 64 * 2)  # BF16
        self.assertEqual(tensor_nbytes((4096,), 24), 4096)  # I8


if __name__ == "__main__":
    unittest.main()
