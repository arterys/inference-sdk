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
from utils import create_folder, DICOM_BINARY_TYPES


from utils import load_image_data, sort_images


def save_secondary_captures(json_response, output_folder_path, multipart_data):
    secondary_capture_parts = [
        p for p in json_response['parts'] if p['binary_type'] in
        ['dicom_structured_report', 'dicom_secondary_capture']
    ]

    # Create DICOM files for secondary capture outputs
    for index in range(len(secondary_capture_parts)):
        file_path = os.path.join(
            output_folder_path, 'sc_{}.dcm'.format(index)
        )
        with open(file_path, 'wb') as outfile:
            outfile.write(multipart_data.parts[index + 1].content)

def upload_study_me(file_path,
                    host,
                    port,
                    output_folder,
                    attachments,
                    override_inference_command='',
                    send_study_size=False,
                    include_label_plots=False,
                    route='/',
                    request_study_path='',
                    encoded_config_xml='',
                    plwmKey=''):
    file_dict = []
    headers = {'Content-Type': 'multipart/related; '}

    images = load_image_data(file_path)
    images = sort_images(images)

    request_json = {
        'studyUID': pydicom.dcmread(images[0].path)['StudyInstanceUID'].value
    }

    if override_inference_command:
        request_json['inference_command'] = override_inference_command

    if encoded_config_xml:
        request_json['encodedConfigXML'] = encoded_config_xml

    if plwmKey:
        request_json['plwmKey'] = plwmKey

    count = 0
    width = 0
    height = 0
    for att in attachments:
        count += 1
        field = str(count)
        fo = open(att, 'rb').read()
        filename = os.path.basename(os.path.normpath(att))
        file_dict.append((field, (filename, fo, 'application/octet-stream')))

    # Either send the path to the study in the request json,
    # or append each dicom file to the request data based on request_study_path flag
    if request_study_path:
        request_json['studyPath'] = request_study_path
        headers['Content-Type'] = 'application/json'
    else:
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

    target = 'http://' + host + ':' + port + route
    print('Targeting inference request to: {}'.format(target))
    if request_study_path:
        r = requests.post(target, json=request_json, headers=headers)
    else:
        file_dict.insert(0, ('request_json', ('request', json.dumps(request_json).encode('utf-8'), 'text/json')))
        me = MultipartEncoder(fields=file_dict)
        boundary = me.content_type.split('boundary=')[1]
        headers['Content-Type'] = headers['Content-Type'] + 'boundary="{}"'.format(boundary)
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

    mask_count = len(json_response["parts"])

    # Assert that we get one binary part for each object in 'parts'
    # The additional two multipart object are: JSON response and request:response digests
    non_buffer_count = 2 if has_digests else 1
    assert mask_count == len(multipart_data.parts) - non_buffer_count, \
        "The server must return one binary buffer for each object in `parts`. Got {} buffers and {} 'parts' objects" \
        .format(len(multipart_data.parts) - non_buffer_count, mask_count)

    masks = [np.frombuffer(p.content, dtype=np.uint8) for i, p in enumerate(multipart_data.parts[1:mask_count+1])
             if json_response['parts'][i]['binary_type'] not in DICOM_BINARY_TYPES]

    if images[0].position is None and \
            all(['dicom_image' in part and 'SOPInstanceUID' in part['dicom_image'] for part in json_response['parts']]):
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

    if len(masks) > 0:
        print("Segmentation mask images generated in folder: {}".format(output_folder))
        print("Saving output masks to files '{}/output_masks_*.npy".format(output_folder))
        for index, mask in enumerate(masks):
            mask.tofile('{}/output_masks_{}.npy'.format(output_folder, index + 1))

    if 'bounding_boxes_2d' in json_response:
        boxes = json_response['bounding_boxes_2d']
        test_inference_boxes.generate_images_with_boxes(images, boxes, output_folder)

    create_folder(output_folder)
    if include_label_plots:
        test_inference_classification.generate_images_with_labels(images, json_response, output_folder)

    with open(os.path.join(output_folder, 'response.json'), 'w') as outfile:
        json.dump(json_response, outfile)

    save_secondary_captures(json_response, output_folder, multipart_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to dicom directory to upload.")
    parser.add_argument("-l", "--include_label_plots", default=False, help="If the model's output are labels and they should be plotted"
        "on top of the .png files.", action='store_true')
    parser.add_argument("--host", default='0.0.0.0', help="Host where inference SDK is hosted")
    parser.add_argument("-p", "--port", default='8900', help="Port of inference SDK host")
    parser.add_argument("-o", "--output", default='output', help="Folder where the script will save the response / output files")
    parser.add_argument('-a', '--attachments', nargs='+', default=[], help='One or more paths to files add as attachments to the request')
    parser.add_argument("-S", "--send_study_size", default=False, help="If the study size should be send in the request JSON",
        action='store_true')
    parser.add_argument("-c", "--inference_command", default='', help="If set, overrides the 'inference_command' send in the request")
    parser.add_argument("-r", "--route", default='/', help="If set, the inference command is directed to the given route. Defaults to '/' route.")
    parser.add_argument("--request_study_path", default='', type=str,
        help="If set, only the given study path is sent to the inference SDK, rather than the study images being sent through HTTP. " \
             "When set, ensure volumes are mounted appropriately in the inference docker container")
    parser.add_argument("-C", "--encoded_config_xml", default='', type=str, help="Optional encoded XML config to be passed as encodedConfigXML in request JSON")
    parser.add_argument("-K", "--plwmKey", default='', type=str, help="base64 encoded license key")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    upload_study_me(args.input,
                    args.host,
                    args.port,
                    args.output,
                    args.attachments,
                    args.inference_command,
                    args.send_study_size,
                    args.include_label_plots,
                    args.route,
                    args.request_study_path,
                    args.encoded_config_xml,
                    args.plwmKey)
