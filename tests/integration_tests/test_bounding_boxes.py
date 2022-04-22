import os
import json
from .mock_server_test_case import MockServerTestCase
from tests.integration_tests.utils import TermColors


class TestBoundingBox(MockServerTestCase):
    input_dir = 'test_box/'
    output_dir = 'test_box_out/'
    command = '-b'
    test_name = 'Bounding box test'

    def testOutputFiles(self):
        self.run_command()

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

        print(TermColors.OKGREEN + "Bounding box test succeeded!!", TermColors.ENDC)
