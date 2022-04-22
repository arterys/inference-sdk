"""
A mock server that uses gateway.py to establish a web server. Depending on the command line options provided,
"-s2D", "-s3D", "-b" or "-cl" the server is capable of returning either a sample 2D segmentation, 3D segmentation,
bounding box or classification labels correspondingly when an inference request is sent to the "/" route.

"""

import argparse
import logging.config
import yaml

import numpy
import pydicom
from utils import tagged_logger

# ensure logging is configured before flask is initialized

with open('logging.yaml', 'r') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('inference')

# pylint: disable=import-error,no-name-in-module
from gateway import Gateway
from flask import make_response


def handle_exception(e):
    logger.exception('internal server error %s', e)
    return 'internal server error', 500


def get_empty_response():
    response_json = {
        'protocol_version': '1.0',
        'parts': []
    }
    return response_json, []


def healthcheck_handler():
    # Return if the model is ready to receive inference requests

    return make_response('READY', 200)


def get_classification_response(dicom_instances):
    dcm = pydicom.read_file(dicom_instances[0])
    response_json = {
        'protocol_version': '1.0',
        'parts': [],
        'series_ml_json': {
            dcm.SeriesInstanceUID: {
                'label': 'healthy',
                'value': '0.9',
                'nested_labels': {
                    'condition1': '0.9',
                    'condition2': '0.2',
                    'condition3': '0.22'
                }
            }
        },
        'study_ml_json': {
            'label1': 'condition1',
            'label2': 'condition2'
        }
    }
    return response_json, []


def get_bounding_box_2d_response(dicom_instances):
    dcm = pydicom.read_file(dicom_instances[0])
    response_json = {
        'protocol_version': '1.0',
        'parts': [],
        'bounding_boxes_2d': [
            {
                'label': 'super bbox',
                'SOPInstanceUID': dcm.SOPInstanceUID,
                'top_left': [5, 5],
                'bottom_right': [10, 10]
            }
        ]
    }
    return response_json, []


def get_probability_mask_3d_response(dicom_instances):
    # Assuming that all files have the same size
    dcm = pydicom.read_file(dicom_instances[0])
    depth = len(dicom_instances)
    image_width = dcm.Columns
    image_height = dcm.Rows
    response_json = {
        'protocol_version': '1.0',
        'parts': [
            {
                'label': 'Mock seg',
                'binary_type': 'probability_mask',
                'binary_data_shape': {
                    'timepoints': 1,
                    'depth': depth,
                    'width': image_width,
                    'height': image_height
                },
                'SeriesInstanceUID': dcm.SeriesInstanceUID
            }
        ]
    }

    array_shape = (depth, image_height, image_width)

    # This code produces a mask that grows from the center of the image outwards as the image slices advance
    mask = numpy.zeros(array_shape, dtype=numpy.uint8)
    mid_x = int(image_width / 2)
    mid_y = int(image_height / 2)
    for s in range(depth):
        offset_x = int(s / depth * mid_x)
        offset_y = int(s / depth * mid_y)
        indices = numpy.ogrid[mid_y - offset_y : mid_y + offset_y, mid_x - offset_x : mid_x + offset_x]
        mask[s][tuple(indices)] = 255

    return response_json, [mask]


def get_probability_mask_2d_response(dicom_instances):
    response_json = {
            'protocol_version': '1.0',
            'parts': []
    }

    masks = []
    for dicom_file in dicom_instances:
        dcm = pydicom.read_file(dicom_file)
        response_json['parts'].append(
            {
                'label': 'Mock seg',
                'binary_type': 'probability_mask',
                'binary_data_shape': {
                    'width': dcm.Columns,
                    'height': dcm.Rows
                },
                'dicom_image': {
                    'SOPInstanceUID': dcm.SOPInstanceUID
                }
            }
        )
        array_shape = (dcm.Rows, dcm.Columns)

        # Generate empty mask (Call your model instead)
        mask = numpy.zeros(array_shape, dtype=numpy.uint8)
        masks.append(mask)

    return response_json, masks


def request_handler_classification(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns labels in free-form json format
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))
    return get_classification_response(dicom_instances)


def request_handler_bbox(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))
    return get_bounding_box_2d_response(dicom_instances)


def request_handler_3d_segmentation(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({ 'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))
    return get_probability_mask_3d_response(dicom_instances)


def request_handler_2d_segmentation(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))
    return get_probability_mask_2d_response(dicom_instances)


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s2D", "--segmentation_model_2D", default=False, help="If the model's output is a 2D segmentation mask",
        action='store_true')
    group.add_argument("-s3D", "--segmentation_model_3D", default=False, help="If the model's output is a 3D segmentation mask",
        action='store_true')
    group.add_argument("-b", "--bounding_box_model", default=False, help="If the model's output are bounding boxes",
        action='store_true')
    group.add_argument("-cl", "--classification_model", default=False, help="If the model's output are labels",
        action='store_true')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    app = Gateway(__name__)
    app.register_error_handler(Exception, handle_exception)
    if args.bounding_box_model:
        app.add_inference_route('/', request_handler_bbox)
    elif args.segmentation_model_3D:
        app.add_inference_route('/', request_handler_3d_segmentation)
    elif args.classification_model:
        app.add_inference_route('/', request_handler_classification)
    else:
        app.add_inference_route('/', request_handler_2d_segmentation)

    app.add_healthcheck_route(healthcheck_handler)
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=True)
