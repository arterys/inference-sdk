import numpy as np
import SimpleITK as sitk
from pydicom import dcmread, dcmwrite
from pydicom.filebase import DicomBytesIO, DicomFileLike
from io import BytesIO

ARTERYS_PROBABILITY_MASK='probability_mask'
ARTERYS_BINARY='binary'
ARTERYS_MULTI_CLASS='multi_class'

def convert_to_nifti(dicom_files, output_file):
    """ This function converts a folder with Dicom files to one nifti file. 
    
    - dicom_files: can be a path to a folder or an array with the dicom files as BytesIO objects.
    """
    
    if isinstance(dicom_files, str):
        # Load files from folder
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_files)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        images_array = np.array([dcmread(DicomBytesIO(dcm.read())).pixel_array for dcm in dicom_files])
        image = sitk.GetImageFromArray(images_array)

    print("Exporting Nifti file of size", image.GetSize())
    sitk.WriteImage(image, output_file)


def load_nifti_file(nifti_file):
    """Read a Nifti file and returns its content as numpy array. """
    nft = sitk.ReadImage(nifti_file)
    return sitk.GetArrayFromImage(nft)


def get_masks_from_nifti_file(nifti_file, data_type=ARTERYS_PROBABILITY_MASK, num_classes=3):
    """Read a Nifti file and returns its content a numpy array segmentation mask which the Arterys viewer supports.
    
    - nifti_file: the path to the Nifti file
    - data_type: Defines how the nifti file contents will be processed. It can be 'probability_mask' (default), 'binary' or 'multi_class'
    - num_classes: if `data_type` is 'multi_class' then pass the number of possible output classes to `num_classes`. (Excluding background)

    The function expects the mask to be in uint8. 
    Returns the segmentation mask as numpy array.
    """

    arr = load_nifti_file(nifti_file)
    if arr.dtype != np.uint8:
        # Only uint8 is supported as segmentation mask output format.
        print("Unsupported output dtype", arr.dtype)
        return None
    
    if data_type == ARTERYS_BINARY:
        arr *= 255
    elif data_type == ARTERYS_MULTI_CLASS:
        output = []
        for label in range(num_classes):
            label_arr = np.copy(arr)
            label_arr[arr == label + 1] = 255
            label_arr[label_arr != 255] = 0
            output.append(label_arr)
        return np.array(output)
    
    return [arr]


def convert_monochrome_1to2(dcm):
    """If the DICOM file `dcm` is in MONOCHROME1 then convert it to MONOCHROME2 """
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        pixels = dcm.pixel_array

        # handle different dtypes
        if pixels.dtype in [np.uint8, np.int8, np.uint16, np.int16]:
            pixels = np.invert(pixels)            
        else:
            raise RuntimeError("Non integer image data types currently not supported.")
        
        # Update original DICOM file
        dcm.PixelData = pixels.tobytes()
        dcm.PhotometricInterpretation = 'MONOCHROME2'
    
    return dcm


def convert_dataset_to_bytes(dataset):
    """ Get a Bytes object from a DICOM dataset.
        Adapted from https://pydicom.github.io/pydicom/dev/auto_examples/memory_dataset.html
    """
    with BytesIO() as buffer:
        memory_dataset = DicomFileLike(buffer)
        dcmwrite(memory_dataset, dataset)
        memory_dataset.seek(0)
        return memory_dataset.read()
