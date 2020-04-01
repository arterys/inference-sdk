"""
Flask HTTP gateway module.

Based off of https://github.com/morpheus-med/vision/blob/master/ml/experimental/research/prod/model_gateway/gateway.py
"""

import functools
from io import BytesIO
import json
import logging
import hashlib

import flask
from flask import Flask, make_response
from utils import request_auditor
from utils import tagged_logger

# pylint: disable=import-error
# Not designed to be installed in vision, yet
from requests_toolbelt import MultipartEncoder, MultipartDecoder

logger = logging.getLogger('gateway')

class InferenceSerializer():
    """Class to convert model outputs to HTTP-friendly binary format.

    Currently, could be a function, but this will likely grow in complexity
    as other response formats are accepted.
    """

    def __call__(self, json_response, binary_components):
        """Generator to convert each part of the model response to text.

        Iterates over the "parts" field of the JSON response and the parts of
        binary_components and converts them to strings.

        :param dict json_response: dictionary of JSON-serializable components
         which describes the binary response format.
        :param list(obj) binary_components: list of binary response components,
         to be serialized to strings by this function
        :return: list(2-tuple(str, str)), one tuple for each binary component,
         where the first string is the HTTP mime-type, and the second string
         is the data of the binary component serialized to a string.
        """

        binary_part_iter = enumerate(
            zip(json_response['parts'], binary_components)
        )
        for i, (json_desc, binary_blob) in binary_part_iter:
            # DEBT: need error handling
            try:
                binary_type = json_desc['binary_type']
            except KeyError:
                logger.error('No binary type for JSON part {}'.format(i))
            except StopIteration:
                logger.error('Ran out of binary components for JSON part {}'.format(i))

            if binary_type in {'png_image'}:
                # Binary blob is assumed to be a file pointer or buffer type
                # to be read directly into the response
                yield ('application/png', binary_blob.read())
            elif binary_type in {'boolean_mask', 'probability_mask'}:
                # Binary blob is a numpy array of any shape
                yield ('application/binary', binary_blob.tostring())


class Gateway(Flask):
    """Main HTTP gateway to receive multipart requests"""

    def __init__(self, *args, **kwargs):
        """Instantiate the model Gateway to delegate to the given function."""
        super().__init__(*args, **kwargs)
        self.add_url_rule('/ping', 'ping', self._pong, methods=['GET', 'POST'])
        self._serializer = InferenceSerializer()
        self._model_routes = {}

    @staticmethod
    def _pong():
        """Handles a ping request with a pong response

        A simple 200. Nothing but the best.
        """

        return make_response('inference-service is up and accepting connections', 200)

    def add_healthcheck_route(self, handler_fn):
        """ Add a handler for the healthcheck route """
        
        self.add_url_rule('/healthcheck', 'healthcheck', handler_fn, methods=['GET', 'POST'])
        
    def add_inference_route(self, route, model_fn):
        """Add a callback function and unique route.

        :param callable model_fn: callback function to use for the backend of
         the provided route.
        :param str route: URL path at which to listen for the route.
        """
        if route in self._model_routes:
            msg = (
                'Route {} already maps to model '.format(route),
                '{}'.format(self._model_routes[route])
            )
            raise ValueError(msg)
        else:
            self._model_routes[route] = model_fn

        logger.info('added inference route %s' % route)

        callback_fn = functools.partial(self._do_inference, model_fn)
        self.add_url_rule(route, route, callback_fn, methods=['POST'])

    def _do_inference(self, model_fn):
        """HTTP endpoint provided by the gateway.

        This function should be partially applied with the model_fn argument
        before it is added as a Flask route.

        Flask functions do not need to take any arguments. They receive the
        request data via the module variable flask.request, which is... somehow
        always supposed to be accurate within the context of a request-handler.

        :param callable model_fn: the callback function to use for inference.
        """
        r = flask.request

        try:
            encoding = r.mimetype_params['charset']
        except KeyError:
            encoding = 'utf-8'

        if not r.content_type.startswith('multipart/related'):
            msg = 'invalid content-type {}'.format(r.content_type)
            logger.error(msg)
            return make_response(msg, 400)

        # Decode JSON and DICOMs into BytesIO buffers and pass to model
        mp = MultipartDecoder(
            content=r.get_data(), content_type=r.content_type,
            encoding=encoding
        )

        input_hash = hashlib.sha256()

        for part in mp.parts:
            input_hash.update(part.content)

        input_digest = input_hash.hexdigest()
        logger.debug('received request with hash %s' % input_digest)

        test_logger = tagged_logger.TaggedLogger(logger)
        test_logger.add_tags({ 'input_hash': input_digest })

        request_json_body = json.loads(mp.parts[0].text)
        request_binary_dicom_parts = [BytesIO(p.content) for p in mp.parts[1:]]

        response_json_body, response_binary_elements = model_fn(
            request_json_body, request_binary_dicom_parts, input_digest
        )

        output_hash = hashlib.sha256()
        output_hash.update(json.dumps(response_json_body).encode('utf-8'))

        for part in response_binary_elements:
            output_hash.update(part)

        output_digest = output_hash.hexdigest()

        test_logger.add_tags({ 'output_hash': output_digest })
        test_logger.debug('request processed')

        logger.debug('sending response with hash %s' % output_digest)

        # Serialize model response to text
        response_body_text_elements = self._serializer(
            response_json_body, response_binary_elements
        )

        # Assemble the list of multipart/related parts
        # The json response must be the first part
        fields = []
        fields.append(
            self._make_field_tuple(
                'json-body', json.dumps(response_json_body),
                content_type='application/json'
            )
        )

        fields.extend(
            self._make_field_tuple('elem_{}'.format(i), elem, mimetype)
            for i, (mimetype, elem) in enumerate(response_body_text_elements)
        )

        fields.append(
            self._make_field_tuple(
                'hashes', input_digest + ':' + output_digest,
                content_type='text/plain'
            )
        )

        # Encode using the same boundary and encoding as original
        encoder = MultipartEncoder(
            fields, encoding=mp.encoding, boundary=mp.boundary
        )

        # Override the Content-Type header that MultipartEncoder uses
        # flask.make_response takes content, response code, and headers
        return make_response(
            encoder.to_string(), 200,
            {'Content-Type': 'multipart/related; boundary={}'.format(mp.boundary)}
        )

    @staticmethod
    def _make_field_tuple(field_name, content_string, content_type,
                          headers=None):
        """Generate a MultipartEncoder field entry.

        MultipartEncoder uses the same syntax as the files argument to
        `requests.post`.

        Requests assumes you want multipart/form-data, and makes certain
        decisions based on that. Namely, you have to provide a field name for
        each "part". You also have to provide a filename for each part. We make
        the field name and filename identical, because we aren't actually
        filling out a form.

        You can provide a dictionary to the files={} argument, and most of the
        requests examples do this. However, dictionaries are unordered; we
        require that field1, field2 be returned to the client in order. To do
        this, we return tuples and assemble a list of parts out of them.

        For more detail, see the Requests documentation on multipart-encoded
        files.

        :param str field_name: name of the form-field for the binary part, and
         also the "filename" of the binary part.
        :param str content_string: string representing the binary content to be
         included in the request
        :param str content_type: string defining the Content-Type (mime type)
         of the request part.
        :param dict(str:str) headers: dictionary of arbitrary HTTP headers and
         header values to include in the request part.
        :return: tuple, suitable for the files argument of `requests.post`. See
         above for details.
        """
        if headers:
            # (filename, data, Content-Type header, other headers)
            content_tuple = (field_name, content_string, content_type, headers)
        else:
            # (filename, data, Content-Type header)
            content_tuple = (field_name, content_string, content_type)

        return (field_name, content_tuple)
