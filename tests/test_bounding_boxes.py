import os
import json
import subprocess
from .mock_server_test_case import MockServerTestCase

class TestBoundingBox(MockServerTestCase):
    input_dir = 'test_box/'
    output_dir = 'test_box_out/'
    command = '-b'
    test_name = 'Bounding box test'

    def testOutputFiles(self):
        input_files = os.listdir(os.path.join('tests', self.input_dir))
        result = subprocess.run(['./send-inference-request.sh', '-b', '--host', '0.0.0.0', '-p',
            '8900', '-o', self.output_dir, self.input_dir], cwd='inference-test-tool',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Test that the command executed successfully
        self.check_success(result, command_name="Send inference request")
        self.assertEqual(result.returncode, 0)

        output_files = os.listdir(os.path.join(self.inference_test_dir, self.output_dir))

        # Test that there was one PNG image generated for each input image
        output_no_index = [name[name.index('_') + 1:] for name in output_files if name.endswith('.png')]
        for name in input_files:
            self.assertTrue((name + '.png') in output_no_index)

        # Test JSON file contents
        file_path = os.path.join(self.inference_test_dir, self.output_dir, 'response.json')
        self.assertTrue(os.path.exists(file_path))
        
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        self.assertIn('protocol_version', data)
        self.assertIn('parts', data)
        self.assertIn('bounding_boxes_2d', data)
        self.assertIsInstance(data['bounding_boxes_2d'], list)

        for box in data['bounding_boxes_2d']:
            self.assertIsInstance(box['label'], str)
            self.assertIsInstance(box['SOPInstanceUID'], str)
            self.assertIsInstance(box['top_left'], list)
            self.assertIsInstance(box['bottom_right'], list)
            self.assertEqual(2, len(box['top_left']))
            self.assertEqual(2, len(box['bottom_right']))

        print("Bounding box test succeeded!!")
