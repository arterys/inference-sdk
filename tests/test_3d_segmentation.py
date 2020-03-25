import os
import json
import subprocess
from .mock_server_test_case import MockServerTestCase

class Test3DSegmentation(MockServerTestCase):
    input_dir = 'test_3d/'
    output_dir = 'test_3d_out/'

    def testOutputFiles(self):
        input_files = os.listdir(os.path.join('tests', self.input_dir))
        result = subprocess.run(['./send-inference-request.sh', '-s', '--host', '0.0.0.0', '-p',
            '8900', '-o', self.output_dir, self.input_dir], cwd='inference-test-tool')
        
        # Test that the command executed successfully
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
        
        for part in data['parts']:
            self.assertIsInstance(part['label'], str)
            self.assertIsInstance(part['binary_type'], str)
            self.assertIn('binary_data_shape', part)
            data_shape = part['binary_data_shape']
            self.assertIsInstance(data_shape['timepoints'], int)
            self.assertIsInstance(data_shape['depth'], int)
            self.assertIsInstance(data_shape['width'], int)
            self.assertIsInstance(data_shape['height'], int)

        # Test if the amount of binary buffers is equals to the elements in `parts`
        output_files = os.listdir(os.path.join(self.inference_test_dir, self.output_dir))
        count_masks = len([f for f in output_files if f.startswith("output_masks_")])
        self.assertEqual(count_masks, len(data['parts']))
