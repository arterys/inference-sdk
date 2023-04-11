"""
A sample request handler for returning inference results as an hdf5 file.
"""

import h5py
import io
import numpy as np

from utils import tagged_logger

# Example processing and inference modules
from src import pre_processing, inference, post_processing


def get_hf5_inference_response(json_input, dicom_instances):
    response_json = {
        "protocol_version": "1.0",
        "parts": [
            {
                "label": "sample",
                "binary_type": "sample_hdf5",
            }
        ],
    }

    # Run Pre-processing
    preprocess_res = pre_processing.request_run(
        dicom_instances, json_input["contour_metadata"], json_input["HeartRate"]
    )

    # Run Inference
    predictions = inference.run(
        input_images=preprocess_res["outimg"], input_meta=preprocess_res["outmeta"]
    )

    # Convert to h5 file bytes for the response
    predictions_h5_bytes = io.BytesIO()
    outh5 = h5py.File(predictions_h5_bytes, "w")

    # Post-processing of data.
    predictions = post_processing.request_run(preprocess_res, predictions)

    create_h5_output(outh5, predictions, preprocess_res)

    return response_json, [predictions_h5_bytes]


def create_h5_output(outh5, predictions, preprocess_res):
    outh5.attrs["slices"] = len(predictions)
    if len(predictions) > 0:
        outh5.attrs["height"] = predictions[0].shape[0]
        outh5.attrs["width"] = predictions[0].shape[1]
        outh5.attrs["timepoints"] = predictions[0].shape[2]
    for slice_index, pred in enumerate(predictions):
        slice_dset = outh5.create_dataset(
            "slice_{}_pred".format(slice_index),
            data=pred,
            dtype=np.float32,
            compression="gzip",
        )

        # One value per entire dataset
        slice_dset.attrs["version"] = 1.0
        slice_dset.attrs["spacing"] = preprocess_res["pixel_spacing"]
        slice_dset.attrs["rotation_angle"] = preprocess_res["rotation_angle"]
        slice_dset.attrs["heart_rate"] = preprocess_res["heart_rate"]
        slice_dset.attrs["image_orientation_patient"] = preprocess_res[
            "image_orientation_patient"
        ]

        # Ordered by Arterys slice_index asc
        slice_dset.attrs["centroid"] = preprocess_res["slice_centroids"][slice_index]
        slice_dset.attrs["first_element_center"] = preprocess_res[
            "first_element_centers"
        ][slice_index]
        slice_dset.attrs["centroid_offset"] = preprocess_res["slice_centroid_offsets"][
            slice_index
        ]
    outh5.close()


def request_handler(json_input, dicom_instances, input_digest):
    """
    Runs inference and returns an output binary part of an hdf5 file
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({"input_hash": input_digest})
    transaction_logger.info("server received json_input len={}".format(len(json_input)))
    return get_hf5_inference_response(json_input, dicom_instances)
