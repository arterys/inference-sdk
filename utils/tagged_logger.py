import logging
import json

class TaggedLogger(logging.LoggerAdapter):
    """
    A tagged logger adapter that allows for persistent tags to be included with every log message.
    """
    def __init__(self, logger):
        """
        Initialize the tagged logger with an empty set of persistent tags.
        """
        self.tags = {}

        if isinstance(logger, TaggedLogger):
            # copy tags from parent tagged logger
            self.tags.update(logger.tags)

        while isinstance(logger, TaggedLogger):
            # find underlying logger instance from parent tagged logger
            logger = logger.logger

        logging.LoggerAdapter.__init__(self, logger, {})

    def process(self, msg, kwargs):
        """
        Format the specified message prefixed by the current persistent tags.
        """
        return "%s - %s" % (json.dumps(self.tags, sort_keys=True), msg), kwargs

    def tags(self):
        """
        Return the current set of persistent tags contained by this logger.
        """
        return self.tags

    def add_tags(self, tags):
        """
        Add new tags to the current set of persistent tags or replace existing tags in the current
        set of persistent tags contained by this logger.
        """
        self.tags.update(tags);

    def tag(self, tags):
        """
        Create a new tagged logger using the current loggers persistent tags with the additional
        specified tags.
        """
        t = TaggedLogger(self)
        t.add_tags(tags);
        return t
