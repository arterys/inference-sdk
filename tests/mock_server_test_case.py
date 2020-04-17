import os
import requests
import subprocess
import sys
import time
import unittest
import shutil
from distutils.dir_util import copy_tree
from .utils import term_colors

class MockServerTestCase(unittest.TestCase):    
    inference_test_dir = 'inference-test-tool'
    input_dir = 'in/'
    output_dir = 'out/'
    command = ''
    test_name = ''
    test_container_name = "arterys_inference_server_tests"
    server_proc = None

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(MockServerTestCase, cls).setUpClass()
        try:
            print("Building test server docker image...")
            proc = subprocess.run(["docker", "build", "-q", "-t", "arterys_inference_server", "."], check=True)
        except:
            print("Failed to build docker image for inference server")
            raise


    def setUp(self):
        print("Starting", self.test_name)
        self.server_proc = subprocess.Popen(["docker", "run", "--rm", "-v", os.getcwd() + ":/opt", "-p", "8900:8000", "--name",
            self.test_container_name, "arterys_inference_server", self.command], stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, encoding='utf-8')

        def cleanup():
            print(term_colors.OKBLUE + "Performing clean up. Stopping inference server...\n", term_colors.ENDC)

            subprocess.run(["docker", "stop", self.test_container_name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(os.path.join(self.inference_test_dir, self.output_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.output_dir))
            if os.path.exists(os.path.join(self.inference_test_dir, self.input_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.input_dir))
            self.server_proc.terminate()
            _, _ = self.server_proc.communicate()

        self.addCleanup(cleanup)
        copy_tree(os.path.join('tests/data', self.input_dir), os.path.join(self.inference_test_dir, self.input_dir))
        self.check_service_up(8900)

    def check_service_up(self, port, endpoint="/", params={}):
        for i in range(30):
            try:
                response = requests.get("http://localhost:{}/healthcheck".format(port))
            except requests.exceptions.ConnectionError:
                pass
            else:
                return response
            time.sleep(1)
        else:
            raise Exception("Service didn't start in time")

    def check_success(self, result, command_name="Subprocess"):
        if result.returncode != 0:
            print(term_colors.FAIL + command_name, "failed with stderr:", term_colors.ENDC)
            print(result.stderr)
            print(term_colors.FAIL + "And stdout:", term_colors.ENDC)
            print(result.stdout)
            self.server_proc.terminate()
            out, err = self.server_proc.communicate()
            print(term_colors.FAIL + "Inference server stderr:", term_colors.ENDC)
            print(err)
            print(term_colors.FAIL + "Inference server stdout:", term_colors.ENDC)
            print(out)
