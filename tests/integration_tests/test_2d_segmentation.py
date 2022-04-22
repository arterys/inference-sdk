import os
import json
import numpy as np
from .mock_server_test_case import MockServerTestCase
from tests.integration_tests.utils import TermColors, DICOM_BINARY_TYPES


class Test2DSegmentation(MockServerTestCase):
    input_dir = 'test_2d/'
    output_dir = 'test_2d_out/'
    command = '-s2D'
    test_name = '2D segmentation test'

    def testOutputFiles(self):
        input_files, output_files = self.run_command()

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
        segmentation_masks_parts = [part for part in data['parts'] if part['binary_type']
               not in DICOM_BINARY_TYPES]
        self.assertEqual(count_masks, len(segmentation_masks_parts))

        for index, part in enumerate(segmentation_masks_parts):
            self.assertIsInstance(part['label'], str)
            self.assertIsInstance(part['binary_type'], str)
            self.assertIn(part['binary_type'], ['heatmap', 'numeric_label_mask', 'dicom_secondary_capture', 'probability_mask', 'boolean_mask'],
                "'binary_type' is not among the supported mask types")
            if part['binary_type'] == 'dicom_secondary_capture' or part['binary_type'] == 'dicom':
                # The rest of the test does not apply
                continue

            self.assertIn('binary_data_shape', part)
            data_shape = part['binary_data_shape']
            self.assertIsInstance(data_shape['width'], int)
            self.assertIsInstance(data_shape['height'], int)
            self.assertIn('dicom_image', part)
            self.assertIsInstance(part['dicom_image']['SOPInstanceUID'], str)

            mask = np.fromfile(os.path.join(self.inference_test_dir, self.output_dir, "output_masks_{}.npy".format(index + 1)), dtype=np.uint8)
            self.assertEqual(mask.shape[0], data_shape['width'] * data_shape['height'])

            if part['binary_type'] == 'heatmap':
                self.validate_heatmap_palettes(part, data)
            elif part['binary_type'] == 'numeric_label_mask':
                self.validate_numeric_label_mask(part, mask)

        print(TermColors.OKGREEN + "2D segmentation test succeeded!!", TermColors.ENDC)
