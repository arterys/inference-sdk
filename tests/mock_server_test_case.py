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
    inference_port = '8900'
    additional_flags = ""

    def setUp(self):
        should_start_server = not os.getenv('ARTERYS_SDK_ASSUME_SERVER_STARTED', False)
        if should_start_server:
            print("Starting", self.test_name)
            self.server_proc = subprocess.Popen(["./start_server.sh", self.command, "--name", self.test_container_name],
                cwd='src', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        else:
            self.inference_port = os.getenv('ARTERYS_SDK_INFERENCE_SERVER_PORT', '8900')
            print("Assuming the server is already running.")

        override_input_folder = os.getenv('ARTERYS_OVERRIDE_TEST_INPUT_FOLDER', "")
        if override_input_folder != "":
            print("Overriding input folder with", override_input_folder)
            self.input_dir = override_input_folder

        self.additional_flags = os.getenv('ARTERYS_TESTS_ADDITIONAL_FLAGS', "")
        healthcheck_method = os.getenv('ARTERYS_SDK_HEALTHCHECK_METHOD', "GET")

        def cleanup():
            print(term_colors.OKBLUE + "Performing clean up. Stopping inference server...\n", term_colors.ENDC)

            self.stop_service()
            if os.path.exists(os.path.join(self.inference_test_dir, self.output_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.output_dir))
            if os.path.exists(os.path.join(self.inference_test_dir, self.input_dir)):
                shutil.rmtree(os.path.join(self.inference_test_dir, self.input_dir))

        self.addCleanup(cleanup)
        copy_tree(os.path.join('tests/data', self.input_dir), os.path.join(self.inference_test_dir, self.input_dir))
        self.check_service_up(self.inference_port, method=healthcheck_method)

    def check_service_up(self, port, method="GET"):
        for i in range(240):
            try:
                if method == "GET":
                    response = requests.get("http://localhost:{}/healthcheck".format(port))
                else:
                    response = requests.post("http://localhost:{}/healthcheck".format(port))
            except requests.exceptions.ConnectionError:
                pass
            else:
                return response
            time.sleep(1)
        else:
            self.stop_service(True)
            raise Exception("Service didn't start in time")

    def stop_service(self, print_output=False):
        if self.server_proc is not None:
            subprocess.run(["docker", "stop", self.test_container_name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.server_proc.terminate()
            out, err = self.server_proc.communicate()
            self.server_proc = None
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

    def validate_heatmap_palettes(self, part, response):
        """ Validates that the heatmap 'part' contains a valid palette

            part: dict with contents of a response 'part'
            response: dict with the whole JSON response from inference server
        """
        if 'palette' in part:
            palette_name = part['palette']
            self.assertIn('palettes', response)
            self.assertIn(palette_name, response['palettes'])
            palette = response['palettes'][palette_name]
            self.assertIn('type', palette)
            self.assertIn('data', palette)
            self.assertIsInstance(palette['data'], list, "'data' must be a list")
            self.assertEqual(palette['type'], 'anchorpoints', "'anchorpoints' is the only supported palette type")
            self.assertGreaterEqual(len(palette['data']), 2, "There must be at least 2 anchorpoints in a 'anchorpoints' palette")
            for ap in palette['data']:
                self.assertIn('threshold', ap, "Anchorpoint must include 'threshold'")
                self.assertIn('color', ap, "Anchorpoint must include 'color'")
                self.assertIsInstance(ap['color'], list, "'color' must be a list")
                self.assertEqual(len(ap['color']), 4, "color must have 4 elements (RGBA)")
                self.assertLessEqual(max(ap["color"]), 255, "Color values must be between 0 and 255")
            self.assertEqual(palette['data'][0]["threshold"], 0.0, "The first anchorpoint must start at 0.0")
            self.assertEqual(palette['data'][-1]["threshold"], 1.0, "The last anchorpoint must end at 1.0")

    def validate_numeric_label_mask(self, part, mask):
        self.assertIn('label_map', part, "A numeric label mask must have a 'label_map' object.")
        label_map = part['label_map']
        labels = label_map.keys()
        for l in labels:
            self.assertTrue(l.isdigit(), "The keys in the 'label_map' must be ints.")
        int_labels = [int(l) for l in labels]
        self.assertLessEqual(mask.max(), max(int_labels), "There are values in the mask which have\
                no associated label from the 'label_map'.")
        str_labels = label_map.values()
        self.assertEqual(len(set(str_labels)), len(str_labels), "The values in 'label_map' must be unique.")
