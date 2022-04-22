import os
import json
from .mock_server_test_case import MockServerTestCase
from tests.integration_tests.utils import TermColors
from jsonschema import validate


class TestClassification(MockServerTestCase):
    input_dir = 'test_classification/'
    output_dir = 'test_classification_out/'
    command = '-cl'
    test_name = 'Classification test'

    def testOutputFiles(self):

        input_files, output_files = self.run_command()

        # If .png files with labels should be generated export the following before
        # running tests: `export ARTERYS_TESTS_ADDITIONAL_FLAGS=-l`
        output_no_index = [name[name.index('_') + 1:] for name in output_files if name.endswith('.png')]

        if '-l' in self.additional_flags or '-include_label_plots' in self.additional_flags:
            for name in input_files:
                self.assertTrue((name + '.png') in output_no_index)

        # Test JSON file contents
        file_path = os.path.join(self.inference_test_dir, self.output_dir, 'response.json')
        self.assertTrue(os.path.exists(file_path))

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        schema = {
            'type' : 'object',
            'required': ['parts', 'protocol_version'],
            'anyOf': [
                {'required': ['study_ml_json']},
                {'required': ['series_ml_json']}
            ],
            'properties' : {
                'protocol_version' : {'type' : 'string'},
                'parts' : {'type' : 'array'},
                'series_ml_json': {
                    'type': 'object'
                },
                'study_ml_json': {
                    'type': 'object'
                }
            },
        }

        validate(data, schema=schema)

        print(TermColors.OKGREEN + "Classification model test succeeded!!", TermColors.ENDC)
