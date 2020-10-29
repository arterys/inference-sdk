import os
import json
import subprocess
import numpy as np
from .mock_server_test_case import MockServerTestCase
from .utils import term_colors

class Test3DSegmentation(MockServerTestCase):
    input_dir = 'test_3d/'
    output_dir = 'test_3d_out/'
    command = '-s3D'
    test_name = '3D segmentation test'

    def testOutputFiles(self):
        input_files = os.listdir(os.path.join('tests/data', self.input_dir))
        result = subprocess.run(['./send-inference-request.sh', '-s', '--host', '0.0.0.0', '-p',
            self.inference_port, '-o', self.output_dir, '-i', self.input_dir] + self.additional_flags.split(),
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
        count_masks = len([f for f in output_files if f.startswith("sc_")])
        secondary_capture_parts = [p for p in data["parts"] if p['binary_type'] == 'dicom_secondary_capture']
        self.assertEqual(count_masks, len(secondary_capture_parts))

        # for index, part in enumerate(data['parts']):
        #     self.assertIsInstance(part['binary_type'], str)
        #     self.assertIn(part['binary_type'], ['heatmap', 'numeric_label_mask', 'dicom_secondary_capture', 'probability_mask', 'boolean_mask'],
        #         "'binary_type' is not among the supported mask types")
        #     if part['binary_type'] == 'dicom_secondary_capture' or part['binary_type'] == 'dicom':
        #         # The rest of the test does not apply
        #         continue


        print(term_colors.OKGREEN + "Secondary capture test succeeded!!", term_colors.ENDC)
