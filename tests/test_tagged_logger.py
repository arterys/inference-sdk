import unittest
import logging

from utils.tagged_logger import TaggedLogger

class TestLogHandler(logging.Handler):
    def emit(self, record):
        self.record = record

class TestTaggedLogger(unittest.TestCase):
    handler = TestLogHandler()
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])
    logger = logging.getLogger("test")

    def setUp(self):
        self.tagged_logger = TaggedLogger(self.logger)

    def assertLevel(self, level):
        self.assertEqual(self.handler.record.levelname, level)

    def assertMessage(self, msg):
        self.assertEqual(self.handler.record.msg, msg)

    def testLogger(self):
        self.logger.warning("this is a warning")
        self.assertLevel("WARNING")
        self.assertMessage("this is a warning")

    def testTaggedLogger(self):
        self.tagged_logger.debug("untagged message")
        self.assertLevel("DEBUG")
        self.assertMessage("{} - untagged message")

    def testAddingTags(self):
        self.tagged_logger.add_tags({ 'a': 1, 'b': 2, 'c': 3 })
        self.tagged_logger.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3} - abc tagged message')

        self.tagged_logger.add_tags({ 'x': 'x', 'y': 'y', 'z': 'z' })
        self.tagged_logger.debug("abcxyz tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3, "x": "x", "y": "y", "z": "z"} - abcxyz tagged message')

    def testOverridingTags(self):
        self.tagged_logger.add_tags({ 'a': 1, 'b': 2, 'c': 3 })
        self.tagged_logger.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3} - abc tagged message')

        self.tagged_logger.add_tags({ 'a': 100, 'b': 200, 'c': 300 })
        self.tagged_logger.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 100, "b": 200, "c": 300} - abc tagged message')

    def testInheritedTags1(self):
        self.tagged_logger.add_tags({ 'a': 1, 'b': 2, 'c': 3 })
        self.tagged_logger.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3} - abc tagged message')

        child = self.tagged_logger.tag({ 'x': 'x', 'y' : 'y', 'z' : 'z' })

        child.debug("abcxyz tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3, "x": "x", "y": "y", "z": "z"} - abcxyz tagged message')

        self.tagged_logger.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3} - abc tagged message')

    def testInheritedTags2(self):
        self.tagged_logger.add_tags({ 'a': 1, 'b': 2, 'c': 3 })
        self.tagged_logger.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3} - abc tagged message')

        child = TaggedLogger(self.tagged_logger)

        child.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3} - abc tagged message')

    def testInheritedTags3(self):
        self.tagged_logger.add_tags({ 'a': 1, 'b': 2, 'c': 3 })
        self.tagged_logger.debug("abc tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3} - abc tagged message')

        parent = self.tagged_logger.tag({ 'x': 'x', 'y' : 'y', 'z' : 'z' })

        parent.debug("abcxyz tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3, "x": "x", "y": "y", "z": "z"} - abcxyz tagged message')

        child = parent.tag({ 'i': 'i', 'j' : 'j', 'k' : 'k' })

        child.debug("abcijkxyz tagged message")
        self.assertLevel("DEBUG")
        self.assertMessage('{"a": 1, "b": 2, "c": 3, "i": "i", "j": "j", "k": "k", "x": "x", "y": "y", "z": "z"} - abcijkxyz tagged message')

if __name__ == "__main__":
    unittest.main()
