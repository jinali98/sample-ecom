from typing import Counter
import unittest

from chat import get_response

class TestGetResponse(unittest.TestCase):

    def test_get_response(self):
        msg = "bla bla bla"
        expected = "Sorry I don't get you!!"
        self.assertEqual(expected, get_response(msg))

    def test_get_response_2(self):
        msg = "hello"
        expected = "Sorry I don't get you!!"
        self.assertEqual(expected, get_response(msg))

    def test_get_response_3(self):
        msg = "hello"
        expected_list = [
        "Hey :-)",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
        result = get_response(msg)
        self.assertTrue(result in expected_list)  