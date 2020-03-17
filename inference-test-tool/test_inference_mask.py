"""
This script lets you test if the inference outputs will be processed correctly by the Arterys server.
"""

import os
import subprocess
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_color_lut, convert_color_space
from utils import load_image_data, sort_images, create_folder

colors = [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]]

def get_colors(index, max_value):
    if index < len(colors):
        return np.array(colors[index]) * max_value
    else:
        rng = np.random.RandomState(index)
        return rng.randint(0, max_value, 3)

def _get_images_and_masks(dicom_images, inference_results):
    if isinstance(dicom_images, str):
        images = load_image_data(dicom_images)
        images = sort_images(images)
    else:
        images = dicom_images

    if isinstance(inference_results, str):
        masks = [np.fromfile(inference_results, dtype=np.uint8)]
    else:
        masks = inference_results
    return (images, masks)

def generate_images_with_masks(dicom_images, inference_results, output_folder):
    """ This function will save images to disk to preview how a mask looks on the input images.
        It saves one image for each input DICOM file. All masks in `inference_results` will be applied to the 
        whole 3D volume of DICOM images. Each mask will show in a different color.
        
        - dicom_images: Array of DCM_Image or path to a folder with images
        - inference_results: Array with mask buffers (one for each image), or path to folder with a numpy file containing one mask.
        - output_folder: Where the output images will be saved 
    """ 
    images, masks = _get_images_and_masks(dicom_images, inference_results)
    create_folder(output_folder)
    
    mask_alpha = 0.5
    offset = 0
    for index, image in enumerate(images):
        dcm = pydicom.dcmread(image.path)
        pixels = _get_pixels(dcm)
        max_value = np.iinfo(pixels.dtype).max

        for mask_index, mask in enumerate(masks):
            # get mask for this image
            image_mask = mask[offset : offset + dcm.Rows * dcm.Columns]
            offset += dcm.Rows * dcm.Columns
            
            pixels = np.reshape(pixels, (-1, 3))
            # apply mask
            pixels[image_mask > 128] = pixels[image_mask > 128] * (1 - mask_alpha) + \
                (mask_alpha * np.array(get_colors(mask_index, max_value)).astype(np.float)).astype(np.uint8)
            
        # write image to output folder
        output_filename = os.path.join(output_folder, str(index) + os.path.basename(os.path.normpath(image.path)))
        output_filename += '.png'
        
        pixels = np.reshape(pixels, (dcm.Rows, dcm.Columns, 3))
        plt.imsave(output_filename, pixels)

def generate_images_for_single_image_masks(dicom_images, inference_results, output_folder):
    """ This function will save images to disk to preview how a mask looks on the input images.
        It saves one image for each input DICOM file with the corresponding `inference_results` mask
        applied as overlay.
        
        - dicom_images: Array of DCM_Image or path to a folder with images
        - inference_results: Array with mask buffers (one for each image)
        - output_folder: Where the output images will be saved 

        The difference with `generate_images_with_masks` is that `generate_images_with_masks` applies each mask to the whole
        volume while this functions applies each mask to one image.
    """
    images, masks = _get_images_and_masks(dicom_images, inference_results)
    create_folder(output_folder)
    
    mask_alpha = 0.5
    for index, (image, mask) in enumerate(zip(images, masks)):
        dcm = pydicom.dcmread(image.path)
        pixels = _get_pixels(dcm)
        max_value = np.iinfo(pixels.dtype).max

        # get mask for this image
        image_mask = mask
        pixels = np.reshape(pixels, (-1, 3))

        # apply mask
        pixels[image_mask > 128] = pixels[image_mask > 128] * (1 - mask_alpha) + \
            (mask_alpha * np.array(get_colors(0, max_value)).astype(np.float)).astype(np.uint8)
            
        # write image to output folder
        output_filename = os.path.join(output_folder, str(index) + os.path.basename(os.path.normpath(image.path)))
        output_filename += '.png'
        
        pixels = np.reshape(pixels, (dcm.Rows, dcm.Columns, 3))
        plt.imsave(output_filename, pixels)
    
def _get_pixels(dicom_file):
    pixels = dicom_file.pixel_array
    if dicom_file.PhotometricInterpretation == 'PALETTE COLOR':
        pixels = apply_color_lut(pixels, dicom_file)
    elif dicom_file.PhotometricInterpretation == 'YBR_FULL_422' or dicom_file.PhotometricInterpretation == 'YBR_FULL':
        pixels = convert_color_space(pixels, dicom_file.PhotometricInterpretation, 'RGB')
        dicom_file.PhotometricInterpretation = 'RGB'
    elif len(pixels.shape) == 2 and (dicom_file.PhotometricInterpretation == 'MONOCHROME1' 
        or dicom_file.PhotometricInterpretation == 'MONOCHROME2'):
        pixels = np.stack((pixels, pixels, pixels), axis=2)

    # handle different dtypes
    if pixels.dtype == np.uint16:
        pixels = (pixels * (256.0 / pixels.max())).astype(np.uint8)
    elif pixels.dtype == np.int16:
        pix_max = pixels.max()
        pix_min = pixels.min()
        pixels = ((pixels + pix_min) * (256.0 / (pix_max - pix_min))).astype(np.uint8)

    return pixels

def _decode(dicom_folder):
    subprocess.check_call(['./decode_dcm.sh', dicom_folder])

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
    generate_images_with_masks(args.file_path, args.inference_results, output_folder)
