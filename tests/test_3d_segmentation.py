import os
import json
import subprocess
import numpy as np
import SimpleITK as sitk
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
            '8900', '-o', self.output_dir, '-i', self.input_dir], cwd='inference-test-tool',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        
        # Test that the command executed successfully
        self.check_success(result, command_name="Send inference request")
        self.assertEqual(result.returncode, 0)

        output_files = os.listdir(os.path.join(self.inference_test_dir, self.output_dir))

        # Test that there is one binary mask saved
        self.assertTrue('output_masks_1.npy' in output_files)

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
        output_folder_path = os.path.join(self.inference_test_dir, self.output_dir)
        output_files = os.listdir(output_folder_path)
        count_masks = len([f for f in output_files if f.startswith("output_masks_")])
        self.assertEqual(count_masks, len(data['parts']))

        for index, part in enumerate(data['parts']):
            self.assertIsInstance(part['label'], str)
            self.assertIsInstance(part['binary_type'], str)
            self.assertIn(part['binary_type'], ['heatmap', 'numeric_label_mask', 'dicom_secondary_capture', 'probability_mask'],
                "'binary_type' is not among the supported mask types")
            if part['binary_type'] == 'dicom_secondary_capture':
                # The rest of the test does not apply
                continue

            self.assertIn('binary_data_shape', part)
            data_shape = part['binary_data_shape']
            self.assertIsInstance(data_shape['timepoints'], int)
            self.assertIsInstance(data_shape['depth'], int)
            self.assertIsInstance(data_shape['width'], int)
            self.assertIsInstance(data_shape['height'], int)

            # test that the mask shape is as advertised
            mask = np.fromfile(os.path.join(self.inference_test_dir, self.output_dir, "output_masks_{}.npy".format(index + 1)), dtype=np.uint8)
            self.assertEqual(mask.shape[0], data_shape['timepoints'] * data_shape['depth'] * data_shape['width'] * data_shape['height'])

            if part['binary_type'] == 'heatmap':
                if 'palette' in part:
                    palette_name = part['palette']
                    self.assertIn('palettes', data)
                    self.assertIn(palette_name, data['palettes'])
                    palette = data['palettes'][palette_name]
                    self.assertIn('type', palette)
                    self.assertIn('data', palette)
                    self.assertIsInstance(palette['data'], list, "'data' must be a list")
                    if palette['type'] == 'lut':
                        self.assertEqual(len(palette['data']), 1028, "LUT tables must have 1028 values (RGBA * 256)")
                    elif palette['type'] == 'anchorpoints':                    
                        self.assertGreaterEqual(len(palette['data']), 2, "There must be at least 2 anchorpoints in a 'anchorpoints' palette")                        
                        for ap in palette['data']:
                            self.assertIn('threshold', ap, "Anchorpoint must include 'threshold'")
                            self.assertIn('color', ap, "Anchorpoint must include 'color'")
                            self.assertIsInstance(ap['color'], list, "'color' must be a list")
                            self.assertEqual(len(ap['color']), 4, "color must have 4 elements (RGBA)")
                            self.assertLessEqual(max(ap["color"]), 255, "Color values must be between 0 and 255")
                        self.assertEqual(palette['data'][0]["threshold"], 0.0, "The first anchorpoint must start at 0.0")
                        self.assertEqual(palette['data'][-1]["threshold"], 1.0, "The last anchorpoint must end at 1.0")
            elif part['binary_type'] == 'numeric_label_mask':
                self.assertIn('label_map', part)

        print(term_colors.OKGREEN + "3D segmentation test succeeded!!", term_colors.ENDC)
