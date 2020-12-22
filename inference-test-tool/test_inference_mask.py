"""
This script lets you test if the inference outputs will be processed correctly by the Arterys server.
"""

import os
import argparse
import random
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from utils import load_image_data, sort_images, create_folder, get_pixels
import cv2

DICOM_BINARY_TYPES = {'dicom_secondary_capture', 'dicom'}

colors = [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]]

def get_colors(index, max_value):
    if index < len(colors):
        arr = np.array(colors[index]) * max_value
    else:
        rng = np.random.RandomState(index)
        arr = rng.randint(0, max_value, 3)

    return np.append(arr, [max_value])

def apply_lut(mask, colormap):
    # colormap is an array of 1024 uint8
    if colormap is None:
        cmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        cmap = np.reshape(cmap, (-1, 3))
        return np.hstack((cmap, np.reshape(np.full(cmap.shape[0], 255, dtype=np.uint8), (-1, 1))))

    lut = np.reshape(colormap, [-1, 4])
    return np.dstack([cv2.LUT(mask, lut[:, i]) for i in range(4)])

def create_lut_from_anchorpoints(anchorpoints):
    assert anchorpoints[0]['threshold'] == 0.0, "The first anchorpoint must have a threshold of 0.0"
    assert anchorpoints[-1]['threshold'] == 1.0, "The last anchorpoint must have a threshold of 1.0"

    lut = []
    for c in range(4):
        channel_arr = []
        for p in range(len(anchorpoints) - 1):
            thr = (anchorpoints[p+1]['threshold'] - anchorpoints[p]['threshold']) * 256

            # Make sure we get 256 values and there is no rounding issue
            if p == len(anchorpoints) - 2:
                thr = 256 - len(channel_arr)

            values = np.linspace(anchorpoints[p]['color'][c],
                anchorpoints[p+1]['color'][c], int(thr), dtype=np.uint8)
            channel_arr.extend(values)
        lut.append(channel_arr)
    lut = np.transpose(np.array(lut))
    return lut

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

def generate_images_with_masks(dicom_images, inference_results, response_json, output_folder):
    """ This function will save images to disk to preview how a mask looks on the input images.
        It saves one image for each input DICOM file. All masks in `inference_results` will be applied to the
        whole 3D volume of DICOM images. Each mask will show in a different color.

        - dicom_images: Array of DCM_Image or path to a folder with images
        - inference_results: Array with mask buffers (one for each image), or path to folder with a numpy file containing one mask.
        - response_json: The JSON response from the inference server
        - output_folder: Where the output images will be saved
    """
    images, masks = _get_images_and_masks(dicom_images, inference_results)
    create_folder(output_folder)

    # Filter out secondary capture outputs
    all_mask_parts = [p for p in response_json["parts"] if p['binary_type'] not in DICOM_BINARY_TYPES]
    secondary_capture_indexes = [i for (i,p) in enumerate(response_json["parts"]) if p['binary_type'] in DICOM_BINARY_TYPES]
    masks = np.array(masks)
    secondary_capture_indexes_bool = np.in1d(range(masks.shape[0]), secondary_capture_indexes)
    secondary_captures = masks[secondary_capture_indexes_bool]
    non_sc_masks = masks[~secondary_capture_indexes_bool]

    # Create DICOM files for secondary capture outputs
    for index, sc in enumerate(secondary_captures):
        dcm = pydicom.read_file(BytesIO(sc.tobytes()))
        file_path = os.path.join(output_folder, 'sc_' + str(index) + '.dcm')
        pydicom.dcmwrite(file_path, dcm)

    offset = 0

    for index, image in enumerate(images):
        dcm = pydicom.dcmread(image.path)
        pixels = get_pixels(dcm)

        # Reshape and add alpha
        pixels = np.reshape(pixels, (-1, 3))
        pixels = np.hstack((pixels, np.reshape(np.full(pixels.shape[0], 255, dtype=np.uint8), (-1, 1))))

        for mask_index, (mask, json_part) in enumerate(zip(non_sc_masks, all_mask_parts)):
            # If the input holds multiple timepoints but the result only includes 1 timepoint
            if image.timepoint is not None and image.timepoint > 0 and json_part['binary_data_shape']['timepoints'] == 1:
                continue
            # get mask for this image
            image_mask = mask[offset : offset + dcm.Rows * dcm.Columns]
            pixels = _draw_mask_on_image(pixels, image_mask, json_part, response_json, mask_index, mask_index)

        offset += dcm.Rows * dcm.Columns

        # write image to output folder
        output_filename = os.path.join(output_folder, str(index) + '_' + os.path.basename(os.path.normpath(image.path)))
        output_filename += '.png'

        if pixels.shape[1] != 4:
            pixels = np.hstack((pixels, np.reshape(np.full(pixels.shape[0], 255, dtype=np.uint8), (-1, 1))))
        pixels = np.reshape(pixels, (dcm.Rows, dcm.Columns, 4))
        plt.imsave(output_filename, pixels)

    for mask_index, mask in enumerate(non_sc_masks):
        assert mask.shape[0] <= offset, "Mask {} does not have the same size ({}) as the volume ({})".format(mask_index, mask.shape[0], offset)

def generate_images_for_single_image_masks(dicom_images, inference_results, response_json, output_folder):
    """ This function will save images to disk to preview how a mask looks on the input images.
        It saves one image for each input DICOM file with the corresponding `inference_results` mask
        applied as overlay.

        - dicom_images: Array of DCM_Image or path to a folder with images
        - inference_results: Array with mask buffers (one for each image)
        - response_json: The JSON response from the inference server
        - output_folder: Where the output images will be saved

        The difference with `generate_images_with_masks` is that `generate_images_with_masks` applies each mask to the whole
        volume while this functions applies each mask to one image.
    """
    images, masks = _get_images_and_masks(dicom_images, inference_results)
    create_folder(output_folder)

    # Filter out secondary capture outputs
    all_mask_parts = [p for p in response_json["parts"] if p['binary_type'] not in DICOM_BINARY_TYPES]
    secondary_capture_indexes = [i for (i,p) in enumerate(response_json["parts"]) if p['binary_type'] in DICOM_BINARY_TYPES]
    masks = np.array(masks)
    secondary_capture_indexes_bool = np.in1d(range(masks.shape[0]), secondary_capture_indexes)
    secondary_captures = masks[secondary_capture_indexes_bool]
    non_sc_masks = masks[~secondary_capture_indexes_bool]

    # Create DICOM files for secondary capture outputs
    for index, sc in enumerate(secondary_captures):
        dcm = pydicom.read_file(BytesIO(sc.tobytes()))
        file_path = os.path.join(output_folder, 'sc_' + str(index) + '.dcm')
        pydicom.dcmwrite(file_path, dcm)

    for index, (image, mask, json_part) in enumerate(zip(images, non_sc_masks, all_mask_parts)):
        dcm = pydicom.dcmread(image.path)
        pixels = get_pixels(dcm)

        # Reshape and add alpha
        pixels = np.reshape(pixels, (-1, 3))
        pixels = np.hstack((pixels, np.reshape(np.full(pixels.shape[0], 255, dtype=np.uint8), (-1, 1))))

        # get mask for this image
        pixels = _draw_mask_on_image(pixels, mask, json_part, response_json, index, 0)

        # write image to output folder
        output_filename = os.path.join(output_folder, str(index) + '_' + os.path.basename(os.path.normpath(image.path)))
        output_filename += '.png'

        pixels = np.reshape(pixels, (dcm.Rows, dcm.Columns, 4))
        plt.imsave(output_filename, pixels)

def _draw_mask_on_image(pixels, image_mask, json_part, response_json, mask_index, label):
    mask_alpha = 0.5
    max_value = np.iinfo(pixels.dtype).max

    assert image_mask.shape[0] == pixels.shape[0], \
        "The size of mask {} ({}) does not match the size of the image ({})".format(mask_index, image_mask.shape[0], pixels.shape[0])

    # apply mask
    if json_part['binary_type'] == 'probability_mask':
        threshold = (json_part['probability_threshold'] * 255) if 'probability_threshold' in json_part else 128
        pixels[image_mask > threshold] = pixels[image_mask > threshold] * (1 - mask_alpha) + \
            (mask_alpha * np.array(get_colors(label, max_value)).astype(np.float)).astype(np.uint8)
    elif json_part['binary_type'] == 'boolean_mask':
        pixels[image_mask != 0] = pixels[image_mask != 0] * (1 - mask_alpha) + \
            (mask_alpha * np.array(get_colors(label, max_value)).astype(np.float)).astype(np.uint8)
    elif json_part['binary_type'] == 'numeric_label_mask':
        for n in range(1, max_value):
            pixels[image_mask == n] = pixels[image_mask == n] * (1 - mask_alpha) + \
                (mask_alpha * np.array(get_colors(n, max_value)).astype(np.float)).astype(np.uint8)
    elif json_part['binary_type'] == 'heatmap':
        if 'palette' in json_part and json_part['palette'] in response_json['palettes']:
            palette = response_json['palettes'][json_part['palette']]
            if palette['type'] == 'anchorpoints':
                lut = create_lut_from_anchorpoints(palette['data'])
                heatmap = apply_lut(image_mask, lut)
            else:
                raise NotImplementedError(f"Palette type {palette['type']} is not implemented. Use 'anchorpoints'")
        else:
            heatmap = apply_lut(image_mask, None)
        heatmap = np.reshape(heatmap, [-1, 4])

        pixels = (pixels * (1 - mask_alpha) + np.reshape(mask_alpha * (heatmap[:, 3] / 255.0), (-1, 1)) * heatmap).astype(np.uint8)
        pixels[:, 3] = 255
    else:
        # Ignoring others like 'dicom_secondary_capture'
        pass

    return pixels
