import unittest

import numpy as np
from src.features import FeatureExtraction


class FeatureExtractionTest(unittest.TestCase):
    def test_tokens_containing_channel_name(self):
        tokens = ["hello", "world"]
        channel_name = "hello"
        result = FeatureExtraction.tokens_containing_channel_name(
            tokens, channel_name)
        self.assertEqual(result.shape, (2, 1))
        np.testing.assert_array_equal(result, np.array([[1], [0]]))

    def test_count_token_occurrences(self):
        tokens = ["hello", "world", "hello"]
        description = "hello world world"
        result = FeatureExtraction.count_token_occurrences(
            tokens, description)
        self.assertEqual(result.shape, (3, 1))
        np.testing.assert_array_equal(result, np.array([[1], [2], [1]]))

    def test_count_token_occurrences_empty(self):
        tokens = ["hello", "world"]
        description = ""
        result = FeatureExtraction.count_token_occurrences(
            tokens, description)
        self.assertEqual(result.shape, (2, 1))
        np.testing.assert_array_equal(result, np.array([[0], [0]]))

    def test_batch_tokens_containing_channel_name(self):
        tokens = [["hello", "world"], ["foo", "bar"]]
        channel_names = ["hello", "foo"]
        result = FeatureExtraction.batch(
            FeatureExtraction.tokens_containing_channel_name, tokens, channel_names)
        self.assertEqual(result.shape, (2, 2, 1))
        np.testing.assert_array_equal(
            result, np.array([[[1], [0]], [[1], [0]]]))

    def test_batch_count_token_occurrences(self):
        tokens = [["hello", "world"], ["foo", "bar"]]
        descriptions = ["hello world world", "foo bar foo"]
        result = FeatureExtraction.batch(
            FeatureExtraction.count_token_occurrences, tokens, descriptions)
        self.assertEqual(result.shape, (2, 2, 1))
        np.testing.assert_array_equal(
            result, np.array([[[1], [2]], [[2], [1]]]))
