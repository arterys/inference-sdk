"""
This script lets you test if the inference outputs (classification labels) will be processed correctly by the Arterys server.
"""

import os
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pydicom
from utils import load_image_data, get_pixels

def generate_images_with_labels(images, json_response, output_folder):
    """ Generates png images with classification labels plotted on top.

    :param array [DCM_Image] images: images to be plotted on 
    :param dict json_response: dict containing the classification labels. It can be freeform, 
        but needs to contain either or both of the following: 
        "study_ml_json": {
            "label1": "xxx", // freeform
        }
        
        or 

        "series_ml_json": {
            "X.X.X.X": { // SeriesInstanceUID
                "label1": "xxx", // freeform
            }, ...
        }

    :param string output_folder: path to output folder
    """

    Y_SPACING = 10
    X_SPACING = 5
    X_INDENT = 5
    study_level_y = 0

    for index, image in enumerate(images):
        dcm = pydicom.dcmread(image.path)
        series_instance_uid = dcm.SeriesInstanceUID
        pixels = get_pixels(dcm)
        pixels = np.reshape(pixels, (dcm.Rows, dcm.Columns, 3))
        pil_image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(pil_image)
        
        if index == 0 and 'study_ml_json' in json_response:
            study_labels = json_response['study_ml_json']
            draw.text((X_SPACING,study_level_y), text='Study level classification prediction:')
            study_level_y += Y_SPACING
            for label, val in study_labels.items():
                draw.text((X_SPACING+X_INDENT, study_level_y), text=(f'{label}: {val}'))
                study_level_y += Y_SPACING
        
        y = study_level_y
        series_labels = {}

        if 'series_ml_json' in json_response and series_instance_uid in json_response['series_ml_json']:
            series_labels = json_response['series_ml_json'][series_instance_uid]

        draw.text((X_SPACING,y), text='Series level classification prediction:')
        y += Y_SPACING
        for label, val in series_labels.items():
            draw.text((X_SPACING+X_INDENT, y), text=(f'{label}: {val}'))   
            y += Y_SPACING
            
        # write image to output folder
        output_filename = os.path.join(output_folder, str(index) + '_' + os.path.basename(os.path.normpath(image.path)))
        output_filename += '.png'
        pil_image.save(output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help="Path to dicom directory to test.")
    parser.add_argument("-r", "--inference_results", help="Results of your inference model. Can be a path to a file or numpy array")
    parser.add_argument("-o", "--output", help="Path to output folder where the tool will leave the generated images.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    output_folder = 'output'
    if args.output:
        output_folder = args.output
    generate_images_with_labels(args.file_path, args.inference_results, output_folder)
