"""Tests for package logger configuration."""

from __future__ import annotations

import logging
import unittest

from xprof.profile_plugin import logging_config


class LoggingConfigTest(unittest.TestCase):

  def test_logger_name_and_level(self):
    self.assertEqual(
        logging_config.logger.name, 'tensorboard.plugins.profile'
    )
    self.assertEqual(logging_config.logger.level, logging.INFO)

  def test_logger_has_handler_and_does_not_propagate(self):
    self.assertTrue(logging_config.logger.handlers)
    self.assertFalse(logging_config.logger.propagate)


if __name__ == '__main__':
  unittest.main()
