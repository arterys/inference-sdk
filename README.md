Arterys Inference SDK <!-- omit in toc --> 
===

Inference model integration SDK

## Contents  <!-- omit in toc --> 

- [Integrating the SDK](#integrating-the-sdk)
  - [Handling an inference request](#handling-an-inference-request)
    - [Standard model outputs](#standard-model-outputs)
    - [Request JSON format](#request-json-format)
  - [Build and run the mock inference service container](#build-and-run-the-mock-inference-service-container)
  - [Logging inside inference service](#logging-inside-inference-service)
  - [Containerization](#containerization)
- [Testing the inference server](#testing-the-inference-server)
  - [To send an inference request to the mock inference server](#to-send-an-inference-request-to-the-mock-inference-server)
  - [Running Unit Tests](#running-unit-tests)

## Integrating the SDK

You should use this SDK to allow the Arterys web app to invoke your model. These sections will help you do so.

### Handling an inference request

`gateway.py` is a helper class that establishes an HTTP endpoint for communication with the Arterys app. It accepts
inference requests in the form of a multipart/related HTTP request. The parts in the multipart request are parsed into
a JSON object and an array of buffers containing contents of input DICOM files. They are in turn passed to a handler
function that you can implement and register with the gateway class. The return values of the handler function are
expected to be a JSON object and an array of buffers (the array could be empty). The gateway class will then package the
returned values into a multipart/related HTTP response sent to the Arterys app.

The following example establishes an inference service that listens on http://0.0.0.0:8000, it exposes an endpoint at
'/' that accepts inference requests. It responses with a 5x5 bounding box annotation on the first DICOM instance in the
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

##### 3D Segmentation masks

For a 3D segmentation producing model, the output is expected to be:

* A single JSON object that describes all the segmentations produced
* One or more binary files that contains each segmentation as a probability mask
  
A sample output of the JSON looks like this:

```json
{ "protocol_version":"1.0",
  "parts": [{"label": "Segmentation #1",
             "binary_type": "probability_mask",
             "binary_data_shape": {"timepoints":1,
                                   "depth":264,
                                   "width":512,
                                   "height":512}
            }]
}
```

Note the “parts” array in the JSON above may contain specs for multiple segmentations. 
In the example above, there’s only one segmentation labelled “Segmentation #1”. 
For every element in the “parts” array, there should be a corresponding binary buffer. 
 
The data format of the probability mask binary buffers is as follows:
 
* Each pixel value is expected to be uint8 (0 to 255), not a float. 
  Value of 0 means a probability of 0, value of 255 means a probability of 1.0 (mapping is linear).
* The order of the pixels is in column-row-slice, order. So if you start reading the binary file from the beginning, you should see the pixels in the following order: [(col0, row0, slice0), (col1, row0, slice0) ... (col0, row1, slice0), (col1, row1, slice0) ... (col0, row0, slice1), (col1, row0, slice1) ...].

##### 2D Segmentation masks

If your model generates a 2D mask, i.e. a mask for a 2D image not a volume of images, then most of the previous section
 still applies with some modifications.

First, your JSON response should look like this:

```json
{ "protocol_version":"1.0",
  "parts": [{"label": "Segmentation #1",
             "binary_type": "probability_mask",
             "binary_data_shape": {"width":512,
                                   "height":512},
             "dicom_image": {
                "SOPInstanceUID": "2.25.336451217722347364678629652826931415692",
                "frame_number": 1,
             }
            }]
}
```

> Note: There is no need to specify `depth` and `timepoint` in `binary_data_shape` but there is a `dicom_image`
> object that allows identifying the image.

> Also, `frame_number` is optional. Can be used for multi-frame instances.

You should still return an array of binary buffers apart from the JSON.
For each input image you should return one item in the `parts` array and one binary buffer (unless there was nothing 
detected for that image).


#### Request JSON format
 The request will have a JSON part with this format:

 ```json
    {
        "inference_command": "string",           
    }
 ```

`inference_command` is only needed if your model can run different types of inference. In that case you can use it to
decide whether to return bounding boxes, segmentation masks, etc.
Possible values are: `get-bounding-box-2d` or `get-probability-mask`.

### Build and run the mock inference service container

```bash
# Start the service
docker-compose up -d

# View the logs
docker-compose logs -f

# Test the service
curl localhost:8900/healthcheck
```

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
- It uses Ubuntu 16.04 as the base image
- It exercises pip install to install Python dependencies listed in the requirements.txt file
- It runs mock_server.py on container startup by virtual of the ENTRYPOINT attribute

Developers should modify the Dockerfile to build the software stack that is required to run their models.

There is a separate Dockerfile in the `inference-test-tool` folder which is used to test the model published on the root docker container.

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
./send-inference-request.sh [-h] [-s] [--host HOST] [-p PORT] /path/to/dicom/files
```

Parameters:
* `-h`: Print usage help
* `-s`: Use it if model is a segmentation model. Otherwise bounding box output will be assumed
* `--host` and `--port`: host and port of inference server

> PNG images will be generated and saved in the `inference-test-tool/output` directory as output of the test tool. 
You can check if the model's output will be correctly displayed on the Arterys web app.

For example, if you have a study whose dicom files you have placed in the `<ARTERYS_SDK_ROOT>/inference-test-tool/study-folder` 
folder, you may send this study to your segmentation model listening on port 8600 on the host OS by running the following 
command in the `inference-test-tool` directory:

```bash
./send-inference-request.sh -s --host 0.0.0.0 --port 8600 ./study-folder/
```

For this to work, the folder where you have your DICOM files (`study-folder` in this case) must be a subfolder of 
`inference-test-tool` so that they will be accessible inside the docker container.

### Running Unit Tests

Run the following command in the root of this repo:
```bash
python3 -m unittest
```
