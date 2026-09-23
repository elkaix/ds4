#!/usr/bin/env python3
"""Unit tests for --q8-gate-up-layers parsing. No llama.cpp required."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen4_exp_convert import TRUNK_LAYERS, parse_layer_spec


class ParseLayerSpecTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_layer_spec(""), frozenset())
        self.assertEqual(parse_layer_spec("  "), frozenset())
        self.assertEqual(parse_layer_spec(None), frozenset())

    def test_single(self):
        self.assertEqual(parse_layer_spec("24"), frozenset({24}))

    def test_list_and_range(self):
        self.assertEqual(
            parse_layer_spec("3,7,11,18,24,31,39,45"),
            frozenset({3, 7, 11, 18, 24, 31, 39, 45}),
        )
        self.assertEqual(parse_layer_spec("0-2,47"), frozenset({0, 1, 2, 47}))

    def test_rejects_mtp_and_oob(self):
        with self.assertRaisesRegex(ValueError, "MTP"):
            parse_layer_spec("48")
        with self.assertRaisesRegex(ValueError, "outside trunk"):
            parse_layer_spec("49")
        with self.assertRaisesRegex(ValueError, "outside trunk"):
            parse_layer_spec("-1")
        with self.assertRaisesRegex(ValueError, "empty layer range"):
            parse_layer_spec("5-1")

    def test_trunk_size(self):
        self.assertEqual(TRUNK_LAYERS, 48)
        self.assertEqual(parse_layer_spec("47"), frozenset({47}))


if __name__ == "__main__":
    unittest.main()
