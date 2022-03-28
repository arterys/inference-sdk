import os
import tempfile
import itertools
from operator import attrgetter
from functools import cmp_to_key
from pathlib import Path
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.pixel_data_handlers.util import apply_color_lut, convert_color_space
import numpy as np
import SimpleITK as sitk
from PIL import Image

DICOM_BINARY_TYPES = {'dicom_secondary_capture', 'dicom', 'dicom_structured_report', 'dicom_gsps'}

class DCM_Image:
    def __init__(self, dcm, path):
        self.instanceUID = dcm.SOPInstanceUID
        self.seriesUID = dcm.SeriesInstanceUID
        self.position = dcm.ImagePositionPatient if 'ImagePositionPatient' in dcm else None
        self.orientation = dcm.ImageOrientationPatient if 'ImageOrientationPatient' in dcm else None
        self.instance_number = dcm.InstanceNumber if 'InstanceNumber' in dcm else None
        self.timepoint = None
        self.path = path

    def direction(self):
        if self.orientation is None:
            return None
        row0 = self.orientation[0:3]
        row1 = self.orientation[3:6]
        return np.cross(row0, row1)


def convert_image_to_dicom(image_file):
    """Read an image file, convert it to Dicom and return the file path"""

    # Load pixel array from image.
    img = Image.open(image_file)
    if ('RGB' == img.mode) or ('RGBA' == img.mode):
        # Assuming greyscale image, keep only one channel.
        pix = np.array(img)[:, :, 0]
    elif 'L' == img.mode:
        # One black and white channel.
        pix = np.array(img)[:, :]
    else:
        raise ValueError('Unhandled Image mode: {}'.format(img.mode))

    # Write pixel array to Dicom file.
    stk = sitk.GetImageFromArray(pix)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    img_basename = os.path.splitext(os.path.basename(image_file))[0] + '_'
    dicom_file = tempfile.NamedTemporaryFile(prefix=img_basename).name + '.dcm'
    writer.SetFileName(dicom_file)
    writer.Execute(stk)

    return dicom_file


def load_image_data(folder):
    # Load all files in 'folder' as DICOM files
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            file_path = str(file_path)

            try:
                dcm = pydicom.dcmread(file_path)
            except InvalidDicomError:
                # File is not a valid Dicom file. Assume it is an
                # image, convert it to Dicom, and retry loading it.
                file_path = convert_image_to_dicom(file_path)
                dcm = pydicom.dcmread(file_path)

            images.append(DCM_Image(dcm, file_path))
        elif os.path.isdir(file_path):
            images.extend(load_image_data(file_path))
    return images


def sort_images(images):
    images_by_series = group_by_series(images)
    result = []
    for series in images_by_series.values():
        sorted_images = _sort_series_images(series)
        result.extend(sorted_images)
    return result

def group_by_series(images):
    series_attr = attrgetter('seriesUID')
    return {k: list(g) for k, g in itertools.groupby(sorted(images, key=series_attr), series_attr)}


def _sort_series_images(images):
    pos = images[0].position
    direction = images[0].direction()
    if pos is None or direction is None:
        return images

    spatial_sorted = sorted(images, key=cmp_to_key(lambda item1, item2: np.dot(np.subtract(np.array(item1.position), np.array(pos)), direction) -
        np.dot(np.subtract(np.array(item2.position), np.array(pos)), direction)))

    timepoints = determine_timepoints(spatial_sorted)
    if timepoints == 1:
        return spatial_sorted

    images_per_timepoint = int(len(spatial_sorted) / timepoints)
    assert len(spatial_sorted) % timepoints == 0, "Series instances {} must be a multiple of timepoints {}" \
        .format(len(spatial_sorted), timepoints)

    for k in range(images_per_timepoint):
        spatial_sorted[k * timepoints : (k+1) * timepoints] = sorted(spatial_sorted[k * timepoints : (k+1) * timepoints], key=attrgetter('instance_number'))

    result = []
    for t in range(timepoints):
        for i in range(images_per_timepoint):
            image = spatial_sorted[i*timepoints+t]
            image.timepoint = t
            result.append(image)
    return result

def filter_mask_parts(response_json):
    return [p for p in response_json["parts"] if p['binary_type'] not in DICOM_BINARY_TYPES]

def filter_masks_by_binary_type(masks, all_mask_parts, response_json):
    # Filters the output masks into binary masks vs dicom data such as SC
    # Returns a tuple of two lists: ([binary masks], [dicom data])
    secondary_capture_indexes = [i for (i,p) in enumerate(response_json["parts"]) if p['binary_type'] in DICOM_BINARY_TYPES]
    masks = np.array(masks)
    secondary_capture_indexes_bool = np.in1d(range(masks.shape[0]), secondary_capture_indexes)
    secondary_captures = masks[secondary_capture_indexes_bool]
    binary_masks = masks[~secondary_capture_indexes_bool]
    return (binary_masks, secondary_captures)

def determine_timepoints(images):
    # Supposing images are sorted by position
    pos = images[0].position
    end_index = 1
    while end_index < len(images) and \
        np.linalg.norm(np.subtract(np.array(images[end_index].position), np.array(pos))) < 1e-3:
        end_index += 1
    return end_index

def get_pixels(dicom_file):
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

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
