import os
import json
import subprocess
import numpy as np
from .mock_server_test_case import MockServerTestCase
from .utils import term_colors

class Test2DSegmentation(MockServerTestCase):
    input_dir = 'test_2d/'
    output_dir = 'test_2d_out/'
    command = '-s2D'
    test_name = '2D segmentation test'

    def testOutputFiles(self):
        input_files = os.listdir(os.path.join('tests/data', self.input_dir))
        result = subprocess.run(['./send-inference-request.sh', '-s', '--host', '0.0.0.0', '-p',
            self.inference_port, '-o', self.output_dir, '-i', self.input_dir] + self.additional_flags.split(),
            cwd='inference-test-tool', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

        # Test that the command executed successfully
        self.check_success(result, command_name="Send inference request")
        self.assertEqual(result.returncode, 0)

        output_files = os.listdir(os.path.join(self.inference_test_dir, self.output_dir))

        # Test that there is one binary mask saved per input image
        for (i, name) in enumerate(input_files):
            self.assertTrue('output_masks_{}.npy'.format(i + 1) in output_files)

        # Test that there was one PNG image generated for each input image
        output_no_index = [name[name.index('_') + 1:] for name in output_files if name.endswith('.png')]
        for name in input_files:
            self.assertTrue((name + '.png') in output_no_index)

        # Test JSON response
        file_path = os.path.join(self.inference_test_dir, self.output_dir, 'response.json')
        self.assertTrue(os.path.exists(file_path))

        with open(file_path) as json_file:
            data = json.load(json_file)

        self.assertIn('protocol_version', data)
        self.assertIn('parts', data)

        # Test if the amount of binary buffers is equals to the elements in `parts`
        output_files = os.listdir(os.path.join(self.inference_test_dir, self.output_dir))
        count_masks = len([f for f in output_files if f.startswith("output_masks_")])
        self.assertEqual(count_masks, len(data['parts']))

        for index, part in enumerate(data['parts']):
            self.assertIsInstance(part['label'], str)
            self.assertIsInstance(part['binary_type'], str)
            self.assertIn(part['binary_type'], ['heatmap', 'dicom_secondary_capture', 'probability_mask', 'boolean_mask'],
                "'binary_type' is not among the supported mask types")
            if part['binary_type'] == 'dicom_secondary_capture':
                # The rest of the test does not apply
                continue
            elif part['binary_type'] == 'heatmap':
                self.validate_heatmap_palettes(part, data)
            self.assertIn('binary_data_shape', part)
            data_shape = part['binary_data_shape']
            self.assertIsInstance(data_shape['width'], int)
            self.assertIsInstance(data_shape['height'], int)
            self.assertIn('dicom_image', part)
            self.assertIsInstance(part['dicom_image']['SOPInstanceUID'], str)

            mask = np.fromfile(os.path.join(self.inference_test_dir, self.output_dir, "output_masks_{}.npy".format(index + 1)), dtype=np.uint8)
            self.assertEqual(mask.shape[0], data_shape['width'] * data_shape['height'])

        print(term_colors.OKGREEN + "2D segmentation test succeeded!!", term_colors.ENDC)
