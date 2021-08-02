"""
This script is meant to mock an inference request from Arterys to your model.
Before executing this script, you must update mock_server.py#request_handler
or implement your own server to direct the request to your model
function (model_fn) and adjust the request literal below as necessary, then
run the container.

Ex. python mock_upload_study.py <path_to_dicom_dir>
"""

import argparse
import os
import requests
import json

from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from base64 import b64encode
from requests_toolbelt import MultipartEncoder
from requests_toolbelt.multipart import decoder
import pydicom
import numpy as np
import test_inference_mask, test_inference_boxes, test_inference_classification
from utils import create_folder


from utils import load_image_data, sort_images

SEGMENTATION_MODEL = "SEGMENTATION_MODEL"
BOUNDING_BOX = "BOUNDING_BOX"
CLASSIFICATION_MODEL = "CLASSIFICATION_MODEL"
OTHER = "OTHER"
ICAD = "ICAD"

def upload_study_me(file_path, model_type, host, port, output_folder, attachments, override_inference_command=None, send_study_size=False, include_label_plots=False, route='/'):
    file_dict = []
    headers = {'Content-Type': 'multipart/related; '}

    images = []
    if file_path:
        images = load_image_data(file_path)
        images = sort_images(images)
    
    if model_type == BOUNDING_BOX:
        print("Performing bounding box prediction")
        inference_command = 'get-bounding-box-2d'
    elif model_type == SEGMENTATION_MODEL:
        if images[0].position is None:
            # No spatial information available. Perform 2D segmentation
            print("Performing 2D mask segmentation")
            inference_command = 'get-probability-mask-2D'
        else:
            print("Performing 3D mask segmentation")
            inference_command = 'get-probability-mask-3D'
    elif model_type == CLASSIFICATION_MODEL:
        print("Performing classification")
        inference_command = 'get-classification-labels'
    else:
        inference_command = 'other'

    if override_inference_command:
        inference_command = override_inference_command

    request_json = {'request': 'post',
                    'route': route,
                    'inference_command': inference_command}

    if model_type == ICAD:
        request_json.studyUID = '1.2.840.113711.676021.3.40956.532496547.26.2116281012.18390'
        request_json.studyPath = 'study'
        request_json.encodedConfigXML = 'PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTE2Ij8+PERpY3Rpb25hcnk+PGl0ZW0+PGtleT48c3RyaW5nPk91dHB1dC5NYW51ZmFjdHVyZU1vZGVsPC9zdHJpbmc+PC9rZXk+PHZhbHVlPjxzdHJpbmc+Jmx0Oz94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTE2Ij8mZ3Q7Jmx0O3N0cmluZyZndDtQb3dlckxvb2stQXJ0ZXJ5cyZsdDsvc3RyaW5nJmd0Ozwvc3RyaW5nPjwvdmFsdWU+PC9pdGVtPjwvRGljdGlvbmFyeT4='

    count = 0
    width = 0
    height = 0
    for att in attachments:
        count += 1
        field = str(count)
        fo = open(att, 'rb').read()
        filename = os.path.basename(os.path.normpath(att))
        file_dict.append((field, (filename, fo, 'application/octet-stream')))

    for image in images:
        try:
            dcm_file = pydicom.dcmread(image.path)
            if width == 0 or height == 0:
                width = dcm_file.Columns
                height = dcm_file.Rows
            count += 1
            field = str(count)
            fo = open(image.path, 'rb').read()
            filename = os.path.basename(os.path.normpath(image.path))
            file_dict.append((field, (filename, fo, 'application/dicom')))
        except:
            print('File {} is not a DICOM file'.format(image.path))
            continue

    print('Sending {} files...'.format(len(images)))
    if send_study_size:
        request_json['depth'] = count
        request_json['height'] = height
        request_json['width'] = width

    file_dict.insert(0, ('request_json', ('request', json.dumps(request_json).encode('utf-8'), 'text/json')))

    me = MultipartEncoder(fields=file_dict)
    boundary = me.content_type.split('boundary=')[1]
    headers['Content-Type'] = headers['Content-Type'] + 'boundary="{}"'.format(boundary)

    target = 'http://' + host + ':' + port + route
    print('Targeting inference request to: {}'.format(target))
    r = requests.post(target, data=me, headers=headers)

    if r.status_code != 200:
        print("Got error status code ", r.status_code)
        exit(1)

    multipart_data = decoder.MultipartDecoder.from_response(r)

    json_response = json.loads(multipart_data.parts[0].text)
    print("JSON response:", json_response)

    last_part = multipart_data.parts[-1]
    has_digests = last_part.headers[b'Content-Type'] == b'text/plain' and \
        len(multipart_data.parts[-1].text) == 129 and multipart_data.parts[-1].text[64] == ':'

    if model_type == SEGMENTATION_MODEL:
        mask_count = len(json_response["parts"])

        # Assert that we get one binary part for each object in 'parts'
        # The additional two multipart object are: JSON response and request:response digests
        non_buffer_count = 2 if has_digests else 1
        assert mask_count == len(multipart_data.parts) - non_buffer_count, \
            "The server must return one binary buffer for each object in `parts`. Got {} buffers and {} 'parts' objects" \
            .format(len(multipart_data.parts) - non_buffer_count, mask_count)

        masks = [np.frombuffer(p.content, dtype=np.uint8) for p in multipart_data.parts[1:mask_count+1]]

        if images[0].position is None:
            # We must sort the images by their instance UID based on the order of the response:
            identifiers = [part['dicom_image']['SOPInstanceUID'] for part in json_response["parts"]]
            filtered_images = []
            for id in identifiers:
                image = next((img for img in images if img.instanceUID == id), None)
                if image:
                    filtered_images.append(image)
            test_inference_mask.generate_images_for_single_image_masks(filtered_images, masks, json_response, output_folder)
        else:
            test_inference_mask.generate_images_with_masks(images, masks, json_response, output_folder)

        print("Segmentation mask images generated in folder: {}".format(output_folder))
        print("Saving output masks to files '{}/output_masks_*.npy".format(output_folder))
        for index, mask in enumerate(masks):
            mask.tofile('{}/output_masks_{}.npy'.format(output_folder, index + 1))
    elif model_type == BOUNDING_BOX:
        boxes = json_response['bounding_boxes_2d']
        test_inference_boxes.generate_images_with_boxes(images, boxes, output_folder)

    elif model_type == CLASSIFICATION_MODEL:
        create_folder(output_folder)
        if include_label_plots:
            test_inference_classification.generate_images_with_labels(images, json_response, output_folder)

    with open(os.path.join(output_folder, 'response.json'), 'w') as outfile:
        json.dump(json_response, outfile)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None, help="Path to dicom directory to upload.")
    parser.add_argument("-s", "--segmentation_model", default=False, help="If the model's output is a segmentation mask",
        action='store_true')
    parser.add_argument("-b", "--bounding_box_model", default=False, help="If the model's output are bounding boxes",
        action='store_true')
    parser.add_argument("-cl", "--classification_model", default=False, help="If the model's output are labels",
        action='store_true')
    parser.add_argument("-l", "--include_label_plots", default=False, help="If the model's output are labels and they should be plotted"
        "on top of the .png files.", action='store_true')
    parser.add_argument("--host", default='0.0.0.0', help="Host where inference SDK is hosted")
    parser.add_argument("-p", "--port", default='8900', help="Port of inference SDK host")
    parser.add_argument("-o", "--output", default='output', help="Folder where the script will save the response / output files")
    parser.add_argument('-a', '--attachments', nargs='+', default=[], help='One or more paths to files add as attachments to the request')
    parser.add_argument("-S", "--send_study_size", default=False, help="If the study size should be send in the request JSON",
        action='store_true')
    parser.add_argument("-c", "--inference_command", default=None, help="If set, overrides the 'inference_command' send in the request")
    parser.add_argument("-r", "--route", default='/', help="If set, the inference command is directed to the given route. Defaults to '/' route.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if args.segmentation_model:
        model_type = SEGMENTATION_MODEL
    elif args.bounding_box_model:
        model_type = BOUNDING_BOX
    elif args.classification_model:
        model_type = CLASSIFICATION_MODEL
    else:
        model_type = OTHER
    upload_study_me(args.input, model_type, args.host, args.port, args.output, args.attachments, args.inference_command, args.send_study_size, args.include_label_plots, args.route)
