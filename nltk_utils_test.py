import unittest
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

class TestNLTKUtils(unittest.TestCase):

    def test_stem(self):
        self.assertEqual(stem("cutting"), "cut")
        self.assertEqual(stem("happening"), "happen")
        self.assertEqual(stem ("playing"), "play")
        self.assertEqual(stem("visited"), "visit")

    def test_stem_2(self):
        word = "cutting"
        expected = "cut"
        self.assertEqual(expected, stem(word))

    def test_tokenize(self):
        self.assertEqual(tokenize("hello world"), ["hello", "world"])
        self.assertEqual(tokenize("hello how are you?"), ["hello", "how", "are", "you", "?"])
        self.assertEqual(tokenize("Hi!"), ["Hi", "!"])
        self.assertEqual(tokenize("Nice to see you!!"), ["Nice", "to", "see", "you", "!", "!"])

    def test_tokenize_2(self):
        sentence = "Hello my name is John"
        expected = ["Hello", "my", "name", "is", "John"]
        self.assertEqual(expected, tokenize(sentence))

    def test_bag_of_words(self):
        np.allclose(bag_of_words(["i", "am"], ["i", "am", "a", "student"]), [1, 1, 0, 0])
        np.allclose(bag_of_words(["hello", "world"], ["hello", "world"]), [1, 1])
        np.allclose(bag_of_words(["hello", "world"], ["hi", "there"]), [0, 0])
        



