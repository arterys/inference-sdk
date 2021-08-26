import os
import json
import subprocess
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from .mock_server_test_case import MockServerTestCase
from .utils import term_colors

from tests.test_secondary_capture import TestSecondaryCapture

class TestCliSecondaryCapture(TestSecondaryCapture):
    input_dir = 'test_cli_secondary_capture/'
    output_dir = 'test_cli_secondary_capture_out/'
    command = '--cli_model'
    test_name = 'CLI Secondary capture test'
