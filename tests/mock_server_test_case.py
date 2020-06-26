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

    def setUp(self):
        should_start_server = not os.getenv('ARTERYS_SDK_ASSUME_SERVER_STARTED', False)
        if should_start_server:
            print("Starting", self.test_name)
            self.server_proc = subprocess.Popen(["./start_server.sh", self.command, "--name", self.test_container_name], stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, encoding='utf-8')
        else:
            print("Assuming the server is already running.")

        override_input_folder = os.getenv('ARTERYS_OVERRIDE_TEST_INPUT_FOLDER', "")
        if override_input_folder != "":
            print("Overriding input folder with", override_input_folder)
            self.input_dir = override_input_folder

        def cleanup():
            print(term_colors.OKBLUE + "Performing clean up. Stopping inference server...\n", term_colors.ENDC)

            self.stop_service()
            if os.path.exists(os.path.join(self.inference_test_dir, self.output_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.output_dir))
            if os.path.exists(os.path.join(self.inference_test_dir, self.input_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.input_dir))

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

    def stop_service(self, print_output=False):
        if self.server_proc is not None:
            subprocess.run(["docker", "stop", self.test_container_name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.server_proc.terminate()
            out, err = self.server_proc.communicate()
            if print_output:
                print(term_colors.FAIL + "Inference server stderr:", term_colors.ENDC)
                print(err)
                print(term_colors.FAIL + "Inference server stdout:", term_colors.ENDC)
                print(out)

    def check_success(self, result, command_name="Subprocess"):
        if result.returncode != 0:
            print(term_colors.FAIL + command_name, "failed with stderr:", term_colors.ENDC)
            print(result.stderr)
            print(term_colors.FAIL + "And stdout:", term_colors.ENDC)
            print(result.stdout)
            self.stop_service(True)
