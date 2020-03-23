"""
This script lets you test if the inference outputs (bounding boxes) will be processed correctly by the Arterys server.
"""

import os
import argparse

import numpy as np
from PIL import Image, ImageDraw
import pydicom
from utils import load_image_data, create_folder, get_pixels

def generate_images_with_boxes(images, boxes, output_folder):
    # Generate images for boxes. `boxes` should be an array of dict
    # Format: {'label': '?', 'SOPInstanceUID': dcm.SOPInstanceUID, 'top_left': [5, 5], 'bottom_right': [10, 10]}
    create_folder(output_folder)

    for index, image in enumerate(images):
        dcm = pydicom.dcmread(image.path)
        pixels = get_pixels(dcm)
        pixels = np.reshape(pixels, (dcm.Rows, dcm.Columns, 3))

        pil_image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(pil_image)
        image_boxes = [box for box in boxes if image.instanceUID == box['SOPInstanceUID']]

        for box in image_boxes:
            # apply box
            ul = box['top_left']
            br = box['bottom_right']
            points = [tuple(ul), (br[0], ul[1]), tuple(br), (ul[0], br[1]), tuple(ul)]
            draw.line(points, fill="red", width=5)
            
            boxes.remove(box)
            
        # write image to output folder
        output_filename = os.path.join(output_folder, str(index) + os.path.basename(os.path.normpath(image.path)))
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
    generate_images_with_boxes(args.file_path, args.inference_results, output_folder)
