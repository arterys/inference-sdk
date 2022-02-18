import os
import json
import subprocess
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from .mock_server_test_case import MockServerTestCase
from .utils import term_colors

class TestSecondaryCapture(MockServerTestCase):
    input_dir = 'test_secondary_capture/'
    output_dir = 'test_secondary_capture_out/'
    command = '-s3D'
    test_name = 'Secondary capture test'

    def testOutputFiles(self):
        input_files = [] # use os.walk to handle nested input folder
        for r, d, f in os.walk(os.path.join('tests/data', self.input_dir)):
            input_files += f
        result = subprocess.run(['./send-inference-request.sh', '--host=0.0.0.0', '--port=' +
            self.inference_port, '--output=' + self.output_dir, '--input=' + self.input_dir] + self.additional_flags.split(),
            cwd='inference-test-tool', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

        # Test that the command executed successfully
        self.check_success(result, command_name="Send inference request")
        self.assertEqual(result.returncode, 0)

        output_files = os.listdir(os.path.join(self.inference_test_dir, self.output_dir))

        # Test JSON response
        file_path = os.path.join(self.inference_test_dir, self.output_dir, 'response.json')
        self.assertTrue(os.path.exists(file_path))

        with open(file_path) as json_file:
            data = json.load(json_file)

        self.assertIn('protocol_version', data)
        self.assertIn('parts', data)

        # Test if the amount of binary buffers is equals to the elements in `parts`
        output_folder_path = os.path.join(self.inference_test_dir, self.output_dir)
        output_files = os.listdir(output_folder_path)
        count_scs = len([f for f in output_files if f.startswith("sc_")])
        secondary_capture_parts = [p for p in data["parts"] if p['binary_type']
                                   in {'dicom_secondary_capture', 'dicom', 'dicom_structured_report', 'dicom_gsps'}]
        self.assertEqual(count_scs, len(secondary_capture_parts))

        # Read and verify output secondary capture dicom files
        for index, sc in enumerate(secondary_capture_parts):
            file_path = os.path.join(output_folder_path, 'sc_' + str(index) + '.dcm')
            try:
                dcm = pydicom.dcmread(file_path)
            except InvalidDicomError:
                self.fail("output dcm file is invalid!")

        print(term_colors.OKGREEN + "Secondary capture test succeeded!!", term_colors.ENDC)
