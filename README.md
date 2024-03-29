Arterys Inference SDK <!-- omit in toc -->
===
[![Build](https://github.com/arterys/inference-sdk/workflows/Build/badge.svg)](https://github.com/arterys/inference-sdk/actions)

The SDK helps you containerize your model into a Flask app with a predefined API to integrate it with the Arterys Marketplace.

## Contents  <!-- omit in toc -->

- [Integrating the SDK](#integrating-the-sdk)
  - [The healthcheck endpoint](#the-healthcheck-endpoint)
  - [Handling an inference request](#handling-an-inference-request)
    - [Standard model outputs](#standard-model-outputs)
      - [Bounding box](#bounding-box)
      - [Classification labels (and other additional information)](#classification-labels-and-other-additional-information)
      - [Segmentation masks](#segmentation-masks)
        - [Probability mask for 3D Series](#probability-mask-for-3d-series)
        - [Boolean mask for 3D Series](#boolean-mask-for-3d-series)
        - [Heatmaps for 3D series](#heatmaps-for-3d-series)
        - [Heatmaps for 2D series (e.g. X-Rays)](#heatmaps-for-2d-series-eg-x-rays)
        - [Numeric label mask for 3D series](#numeric-label-mask-for-3d-series)
      - [Linear measurements](#linear-measurements)
      - [Secondary capture support](#secondary-capture-support)
      - [DICOM structured report](#dicom-structured-report)
      - [Returning DICOM conformance errors](#returning-dicom-conformance-errors)
    - [Request JSON format](#request-json-format)
  - [Build and run the mock inference service container](#build-and-run-the-mock-inference-service-container)
    - [Adding GPU support](#adding-gpu-support)
  - [Logging inside inference service](#logging-inside-inference-service)
  - [Containerization](#containerization)
- [Testing the inference server](#testing-the-inference-server)
  - [To send an inference request to the mock inference server](#to-send-an-inference-request-to-the-mock-inference-server)
    - [Sending attachments in the requests to the inference server](#sending-attachments-in-the-requests-to-the-inference-server)
  - [Running Unit Tests](#running-unit-tests)
- [Nifti image format support](#nifti-image-format-support)


## Integrating the SDK

You should use this SDK to allow the Arterys web app to invoke your model.
The `gateway.py` is a helper class that creates a Flask server to communicate with the Arterys app via HTTP.

You will have to provide 2 endpoints:

* `GET /healthcheck`: to tell whether the server is ready to handle requests
* `POST /`: to handle inference requests

### The healthcheck endpoint

The Arterys app relies on the result from this endpoint to decide whether or not the inference service is ready to field requests.
You should handle healthcheck requests by returning a string 'READY' if your server is ready.
Otherwise return something else, with status code 200 in both cases.
Returning "READY" too early will result in failures due to requests being sent too early.

You can do this by modifying the `healthcheck_handler` function in `mock_server.py`

### Handling an inference request

The Flask server defined in `gateway.py` accepts inference requests in the form of a multipart/related HTTP request.
The parts in the multipart request are parsed into
a JSON object and an array of buffers containing contents of input DICOM files. They are in turn passed to a handler
function that you can implement and register with the gateway class. The return values of the handler function are
expected to be a JSON object and an array of buffers (the array could be empty). The gateway class will then package the
returned values into a multipart/related HTTP response sent to the Arterys app.

The following example establishes an inference service that listens on http://0.0.0.0:8000, it exposes an endpoint at
'/' that accepts inference requests. It responds with a 5x5 bounding box annotation on the first DICOM instance in the
list of dicom instances it receives.

```
from gateway import Gateway

def handle_exception(e):
    logger.exception('internal server error %s', e)
    return 'internal server error', 500

def handler(json_input, dicom_instances, input_hash):
    logger = tagged_logger.TaggedLogger(logger)
    logger.add_tags({ 'input_hash': input_digest })
    logger.info('mock_model received json_input={}'.format(json_input))

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

if __name__ == '__main__':
    app = Gateway(__name__)
    app.register_error_handler(Exception, handle_exception)
    app.add_inference_route('/', handler)
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=True)
```

See mock_server.py, for more examples of handlers that respond with different types of annotations.

> You will normally not need to change `gateway.py`.
However, you will have to **parse the request, call your model and return a response** in the `mock_server.py`

#### Standard model outputs

##### Bounding box

For a bounding box producing model, the output is expected to be a single JSON object.
Note the bounding_boxes_2d array could be extended to contain multiple entries.
The top_left and bottom_right coordinates are in pixel space (column, row).

For example:

```json
{ "protocol_version":"1.0",
  "bounding_boxes_2d": [{ "label": "Lesion #1",
                          "SOPInstanceUID": "2.25.336451217722347364678629652826931415692",
                          "top_left": [102, 64],
                          "bottom_right": [118, 74]
                          }]
}
```

##### Classification labels (and other additional information)

Classification labels or any other information for the study or series of the input, which you want to include in the result,
can be sent using `study_ml_json` or `series_ml_json` keys.
Both these keys accept freeform JSON and its content will be shown as-is to the end users.
For *series*, add the additional information nested under the appropiate SeriesInstanceUID.

For example:

```jsonc
{
    "protocol_version": "1.0",
    "parts": [{...}], // unchanged
    "study_ml_json": {
      "label1": "xxx", // freeform
    },
    "series_ml_json": {
      "X.X.X.X": { // SeriesInstanceUID
        "label1": "xxx", // freeform
      },
   }
}
```

##### Segmentation masks

For a segmentation producing model, the output is expected to be:

* A single JSON object that describes all the segmentations produced
* One or more binary files that contains each segmentation mask

A sample output of the JSON looks like this:

```json
{ "protocol_version":"1.0",
  "parts": [{"label": "Segmentation #1",
             "binary_type": "probability_mask",
             "binary_data_shape": {"timepoints":1,
                                   "depth":264,
                                   "width":512,
                                   "height":512},
             "SeriesInstanceUID": "1.1.1.1"
            }]
}
```

> The `SeriesInstanceUID` object allows identifying the series to which the result applies to.

Note the “parts” array in the JSON above may contain specs for multiple segmentations.
In the example above, there’s only one segmentation labelled “Segmentation #1”.
For every element in the “parts” array, there should be a corresponding binary buffer.

> The above example is for masks that apply to 3D series. For 2D series you do not need to specify "depth" nor "timepoints"

There are multiple possible interpretations for segmentation masks supported by the SDK, explained in the next subsections:

* Probability mask
* Boolean mask
* Heatmap for 3D Series
* Heatmap for 2D Series (e.g. X-Rays)
* Numeric label mask

Depending on what the mask represents you must specify a different value for `binary_type`.

The data format of the segmentation mask binary buffers should respect the following:

* Each pixel value is expected to be uint8 (0 to 255), not a float.
* The order of the pixels is in column-row-slice, order. So if you start reading the binary file from the beginning, you should see the pixels in the following order: [(col0, row0, slice0), (col1, row0, slice0) ... (col0, row1, slice0), (col1, row1, slice0) ... (col0, row0, slice1), (col1, row0, slice1) ...].


###### Probability mask for 3D Series

To handle probability masks for 3D Series follow the steps for [segmentation masks](#segmentation-masks)
and use a `binary_type` of "probability_mask".

Also, you can optionally add a "probability_threshold" specifying the threshold to show or hide the mask.

A sample output of the JSON looks like this:

```json
{ "protocol_version":"1.0",
  "parts": [{"label": "Segmentation #1",
             "binary_type": "probability_mask",
             "probability_threshold": 0.5,
             "binary_data_shape": {"timepoints":1,
                                   "depth":264,
                                   "width":512,
                                   "height":512},
             "SeriesInstanceUID": "1.1.1.1"
            }]
}
```

The data format of the probability mask binary buffers is as follows:

* Each pixel value is expected to be uint8 (0 to 255), not a float.
  Value of 0 means a probability of 0, value of 255 means a probability of 1.0 (mapping is linear).

###### Boolean mask for 3D Series

Boolean masks are actually a subtype of probability masks.
When you specify a `binary_type` of `boolean_mask` then every non-zero value of the mask will be taken as positive.
It has the same effect as a probability mask with `probability_threshold` set to something like 0.003.

Follow the steps mentioned for probability masks but set `binary_type` to `boolean_mask` if your model returns a boolean mask.

###### Heatmaps for 3D series

To handle heatmaps of 3D volumes follow the steps for [segmentation masks](#segmentation-masks) with the `binary_type` set to `'heatmap'`.

Optionally you can specify custom color palettes and assign them to specific parts:

```jsonc
{ "protocol_version":"1.0",
  "parts": [{"label": "Segmentation #1",
             "binary_type": "heatmap",
             "binary_data_shape": {"timepoints":1,
                                   "depth":264,
                                   "width":512,
                                   "height":512},
             "SeriesInstanceUID": "1.1.1.1",
             "palette": "my_super_palette"
            }],
  "palettes": {
    "my_super_palette": {
      "type": "anchorpoints",
      "data": [
        { "threshold": 0.0, "color": [0, 0, 0, 0] },
        { "threshold": 1.0, "color": [255, 0, 0, 255] }
      ]
    }
  }
}
```

Currently, the only supported palette type is "anchorpoints" as shown in the example above.

###### Heatmaps for 2D series (e.g. X-Rays)

If your model generates a 2D mask, i.e. a mask for a 2D image not a volume of images, then most of the
[segmentation masks](#segmentation-masks) section and [Heatmaps for 3D Series](#heatmaps-for-3d-series)
still applies with some modifications.

First, your JSON response should look like this, including `binary_type` = 'heatmap':

```jsonc
{ "protocol_version":"1.0",
  "parts": [{"label": "Segmentation #1",
             "binary_type": "heatmap",
             "binary_data_shape": {"width":512,
                                   "height":512},
             "dicom_image": {
                "SOPInstanceUID": "2.25.336451217722347364678629652826931415692",
                "frame_number": 1,
             },
             "palette": "my_super_palette"
            }],
  "palettes": {
    "my_super_palette": { ... }
  }
}
```

See [Heatmaps for 3D Series](#heatmaps-for-3d-series) for an example on how to specify `palettes`.

> Note: There is no need to specify `depth` and `timepoints` in `binary_data_shape` but there is a `dicom_image`
> object that allows identifying the image.

> Also, `frame_number` is optional. Can be used for multi-frame instances.

You should still return an array of binary buffers apart from the JSON.
For each input image you should return one item in the `parts` array and one binary buffer (unless there was nothing
detected for that image).

###### Numeric label mask for 3D series

If your model creates segmentations for multiple classes/labels which do not overlap then you should follow the guide for
[segmentation masks](#segmentation-masks) with the following changes:

* The `binary_type` should be `numeric_label_mask`
* The pixel values should still be `uint8`, but the value won't be a probability but the index of the predicted label.
* You should add a key `label_map` with a dictionary which maps the label index to its name.

For example:

```json
{ "protocol_version":"1.0",
  "parts": [{"label": "Segmentation #1",
             "binary_type": "numeric_label_mask",
             "binary_data_shape": {"timepoints":1,
                                   "depth":264,
                                   "width":512,
                                   "height":512},
             "SeriesInstanceUID": "1.1.1.1",
             "label_map": { "1": "Pneumonia",
                            "2": "Healthy",
                            "3": "Covid"}
            }]
}
```

##### Linear measurements

If your model outputs linear measurements you can send those results in this format:

```jsonc
{
    "protocol_version": "1.0",
    "parts": [{...}], // unchanged
    "linear_measurements_2d": [
        {
            "SOPInstanceUID": "...",
            "frame_number": 0, // optional for multi-frame instances
            "label": "...",
            "coordinates_type": "pixel", // valid values are "world" and "pixel"
            "start": [x, y], // float values (can be integers for pixel coordinates).
            "end": [x, y]
        }
    ]
}
```


##### Secondary capture support

If your model's output is a secondary capture DICOM file and you want to return that as result then you have to specify `"binary_type": "dicom_secondary_capture"` in the corresponding `parts` object. Your response could look like this:

```json
{
  "protocol_version": "1.0",
  "parts": [
      {
          "label": "Mock seg",
          "binary_type": "dicom_secondary_capture",
          "SeriesInstanceUID": "X.X.X.X"
      }
  ]
}
```

You should return the secondary capture as a byte stream in your handler.
For an example, the `write_dataset_to_bytes` function on [this Pydicom help page](https://pydicom.github.io/pydicom/stable/auto_examples/memory_dataset.html) might be helpful.

##### DICOM structured report

If your model returns a DICOM Structured Report then do the same as for secondary captures explained in the previous section, just change `'binary_type'` to `'dicom'`.

##### Returning DICOM conformance errors

If you run validations on the input DICOM data to check whether it is compliant with the requirements of your model, you can return the results of these validations with the results from your model.
This is useful if the results you return are sub-optimal for certain reason.

For each validation you can return a description and whether it is compliant or not.
If a `recommended` validation fails then the results will be presented with an informative warning:


```jsonc
{
  "protocol_version": "1.0",
  "parts": [
      {
          "label": "Mock seg",
          "binary_type": "dicom_secondary_capture",
          "SeriesInstanceUID": "X.X.X.X",
          "dicomConformance": {
            "recommended": [
                {
                    "description": "FoV above 170mm",
                    "compliant": false
                },
                ...
            ]
        }
      }
  ]
}
```

#### Request JSON format

Currently the request JSON will not have any meaningful information for you.
However, Arterys could potentially send you important parameters if your model requires any.
This would need custom work from the Arterys support team.

### Build and run the mock inference service container

```bash
# Start the service.
./start_server.sh <command>

# View the logs
docker logs -f <name of the container>

# Test the service
curl localhost:8900/healthcheck
```

If you want to pass additional flags to the `docker run` command which is run in `start_server.sh` then you can pass all of them behind the `command`.

For example, to run the container in background, and add access to GPU:

```bash
./start_server.sh -d --gpus=all
```

While developing it might also be handy to add a volume with the current directory to speed up the test cycle.
To do this add `-v $(pwd):/opt` at the end of the previous command

#### Adding GPU support

If you need GPU support for running your model then you can pass an argument to the `start_server.sh` script. Add `--gpus=all` if your Docker version is >=19.03 or `--runtime=nvidia` if it is <19.03.
You must also have the nvidia-container-toolkit or nvidia-docker installed.

For more information about supporting GPU acceleration inside docker containers please check the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) repository.

### Logging inside inference service

To help with diagnosing problems with the inference service efficiently, it is recommend that logging be added to
trace key stages in the processing of inference requests, as well as to measure performance metrics.

A logging.yaml configuration file is provided, so log messages are formatted in a way that is consumable by the
Arterys logging infrastructure. To use:

```
with open('logging.yaml', 'r') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('inference')
logger.info('hello world')
```

A tagged logger is available to provide useful context to each log message. One useful example is to create
a tagged logger that is tagged with the digest of an incoming inference request. All subsequent messages logged with
this logger will have the input digest attached, which is useful for finding log messages corresponding to a specific
request.

```
from utils import tagged_logger

input_hash = hashlib.sha256()
for dicom_file in dicom_files:
    input_hash.update(dicom_file)

test_logger = tagged_logger.TaggedLogger(logger)
test_logger.add_tags({ 'input_hash': input_hash.hexdigest() })
test_logger.info('start processing')
```

The input_hash is calculated by gateway.py for every transaction, and it is passed to the custom handler.

### Containerization

The default Dockerfile in this repository has the following characteristics:
- It uses `arterys/inference-sdk-base` as the base image
- It exercises pip install to install Python dependencies listed in the requirements.txt file
- It runs mock_server.py on container startup by virtual of the ENTRYPOINT attribute

Developers should modify the Dockerfile to build the software stack that is required to run their models.

While it is not necessary, we recommend using `arterys/inference-sdk-base` as base image repository for your Dockerfile.
There are several tags you can choose from depending on your need of GPU support or not.
All these images will have the basic dependencies that you need as well as a validated CUDA set up if you need it.

You can choose from the following tags (where `<version_number>` is the version of the docker image):

* <version_number>-cpu
* <version_number>-cuda-9.2
* <version_number>-cuda-10.0
* <version_number>-cuda-10.1
* <version_number>-cuda-10.2

Check the [arterys/inference-sdk-base](https://hub.docker.com/r/arterys/inference-sdk-base) repository on DockerHub for the latest published version.
You can also find more information on the [inference-sdk-images](https://github.com/arterys/inference-sdk-images) GitHub repo.

## Testing the inference server

### To send an inference request to the mock inference server

The `inference-test-tool/send-inference-request` script allows you to send dicom data to the mock server and exercise it.
To use it run from inside the `inference-test-tool` folder:

```bash
./send-inference-request.sh <arguments>
```

If you don't specify any arguments, a usage message will be shown.

The script accepts the following parameters:

```
./send-inference-request.sh [-h] [-i INPUT] [-l] [--host HOST] [-p PORT] [-o OUTPUT]
              [-a ATTACHMENTS [ATTACHMENTS ...]] [-S] [-c INFERENCE_COMMAND]
              [-r ROUTE] [--request_study_path REQUEST_STUDY_PATH]
              [--request_options KEY=VALUE [KEY=VALUE ...]]
```

Parameters:
* `-h`: Print usage help
* `-l`: Use it if you want to generate PNG images with labels plotted on top of them (only applies to classification models)
* `--host` and `--port`: host and port of inference server
* `-i`: Input files
* `-a`: Add attachments to the request. Arguments should be paths to files.
* `-S`: If the study size should be send in the request JSON
* `-c`: If set, overrides the 'inference_command' send in the request
* `--request_study_path`: If set, only the given study path is sent to the inference SDK, rather than the study images being sent through HTTP. When set, ensure volumes are mounted appropriately in the inference docker container
*  `--request_options`: Set a number of key-value pairs to be sent in the request JSON (do not put spaces before or after the = sign).
                        If a value contains spaces, you should define it with double quotes.
                        Values are always treated as strings.
                        e.g. --request_options foo=bar a=b greeting="hello there"

> PNG images will be generated and saved in the `inference-test-tool/output` directory as output of the test tool.
You can check if the model's output will be correctly displayed on the Arterys web app.
Classification models allow for free-form json reponses and therefore do not have a single standard of display in the Arterys web app. However, generating
PNG images for classification models may assist in checking which label was generated by the model for which DICOM.

For example, if you have a study whose dicom files you have placed in the `<ARTERYS_SDK_ROOT>/inference-test-tool/study-folder`
folder, you may send this study to your segmentation model listening on port 8900 on the host OS by running the following
command in the `inference-test-tool` directory:

```bash
./send-inference-request.sh --host 0.0.0.0 --port 8900 -i ./study-folder/
```

For this to work, the folder where you have your DICOM files (`study-folder` in this case) must be a subfolder of
`inference-test-tool` so that they will be accessible inside the docker container.

#### Sending attachments in the requests to the inference server

If you need additional files to be sent with each request, such as a license file for example, then those files will be sent as multipart files.
In the `mock_server.py` you will receive a JSON input and a list of files, as usual.
The list of files will include tha attachments at the front followed by the DICOM files to process.

To test this with the test tool you can use the `-a` parameter to add any number of attachments like this:

```bash
./send-inference-request.sh --host 0.0.0.0 -p 8900 -i in/ -a some_attachment.txt other_attachment.bin
```


### Running Unit Tests

Run the following command in the root of this repo:
```bash
python3 -m unittest
```

You must have Python 3.6+ installed.

Currently most tests are actually integration tests.
There is one test for each: 3D series segmentation, 2D series segmentation and bounding box detection.

If you want to run only one of them, run:

```bash
python3 -m unittest tests/<name_of_file>
```
The tests will start the inference server using the `start_server.sh` script.
The server must be stopped before running the tests.
If you want to start the inference server differently then start it before running the tests and define the env variable 'ARTERYS_SDK_ASSUME_SERVER_STARTED=true'.

If you want to use a custom test study then save it in a folder under `tests/data` and define the env var 'ARTERYS_OVERRIDE_TEST_INPUT_FOLDER=<folder_name>'.

In case you want to use any other flag available when running `send-inference-request.sh` you can set `ARTERYS_TESTS_ADDITIONAL_FLAGS` to all the additional flags you want to pass to that command.
This will be appended to the call to `send-inference-request.sh`.
For example:

```
ARTERYS_TESTS_ADDITIONAL_FLAGS="-S -a attachment1.txt"
```


## Nifti image format support

In the `utils/image_conversion.py` there are a few functions that can be helpful if your model accepts Nifti files as input or generates Nifti output files.

To convert Dicom files to Nifti use `convert_to_nifti`. If you want to load a segmenation mask from a Nifti file you can use `get_masks_from_nifti_file`.
