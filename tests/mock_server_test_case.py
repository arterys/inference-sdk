import os
import requests
import subprocess
import sys
import time
import unittest
import shutil
from distutils.dir_util import copy_tree

class MockServerTestCase(unittest.TestCase):    
    inference_test_dir = 'inference-test-tool'
    input_dir = 'in/'
    output_dir = 'out/'
    command = ''
    test_name = ''
    test_container_name = "arterys_inference_server_tests"

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
        proc = subprocess.run(["docker", "run", "--rm", "-v", os.getcwd() + ":/opt", "-p", "8900:8000", "--name",
            self.test_container_name, "-d", "arterys_inference_server", self.command], stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, check=True)

        def cleanup():
            print("Performing clean up. Stopping inference server...\n")
            subprocess.run(["docker", "stop", self.test_container_name],
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

    def check_success(self, result, command_name="Subprocess"):
        if result.returncode != 0:
            out, err = result.communicate()
            print(command_name, "failed with error:")
            print(err, "\nAnd output:\n", out)
