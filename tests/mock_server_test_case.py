import os
import requests
import subprocess
import sys
import time
import unittest
import shutil
from distutils.dir_util import copy_tree

class MockServerTestCase(unittest.TestCase):
    compose_dir = './'
    inference_test_dir = 'inference-test-tool'
    input_dir = 'in/'
    output_dir = 'out/'

    def setUp(self):
        compose_file = os.path.join(
            os.path.dirname(__file__), self.compose_dir,
            "docker-compose.yml")
        proc = subprocess.Popen(["docker-compose", "up"])

        def cleanup():
            subprocess.call(["docker-compose", "down"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(os.path.join(self.inference_test_dir, self.output_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.output_dir))
            if os.path.exists(os.path.join(self.inference_test_dir, self.input_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.input_dir))

        self.addCleanup(cleanup)
        copy_tree(os.path.join('tests', self.input_dir), os.path.join(self.inference_test_dir, self.input_dir))
        self.check_service_up(8900)

    def check_service_up(self, port, endpoint="/", params={}):
        for i in range(5):
            try:
                response = requests.get("http://localhost:{}/healthcheck".format(port))
            except requests.exceptions.ConnectionError:
                pass
            else:
                return response
            time.sleep(1)
        else:
            raise Exception("Service didn't start in time")
