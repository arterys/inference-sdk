"""
Module for logging inference request and response information to S3 in a hashed de-identified form,
for auditing and traceability.
"""

import boto3
import hashlib
import json
import logging
import os

from utils import tagged_logger
from random import choice
from string import ascii_uppercase

S3_AUDIT_BUCKET_NAME=os.environ['S3_AUDIT_BUCKET_NAME']

logger = logging.getLogger('request_auditor')

def write_s3_audit(audit_info):
    """
    Writes an audit message to s3 of an inference request and response hash. Expects a dictionary of the form:
        {
            "input_hash": <sha256 hash string of the input data>,
            "output_hash": <sha256 hash string of the output data>,
            "vendor": <The name of the organization providing the inference model>
        }

    This function should be called on every inference request.
    """

    vendor = audit_info['vendor'] if 'vendor' in audit_info else 'Unknown'

    audit_logger = tagged_logger.TaggedLogger(logger)
    audit_logger.add_tags({ 'vendor': vendor })
    audit_logger.add_tags({ 'input_hash': audit_info['input_hash'] })
    audit_logger.add_tags({ 'output_hash': audit_info['output_hash'] })

    s3 = boto3.resource('s3')
    object_key = '{}:{}'.format(audit_info['input_hash'], audit_info['output_hash'])
    object_tag = 'Vendor={}'.format(vendor)

    try:
        s3.Bucket(S3_AUDIT_BUCKET_NAME).put_object(Key=object_key, Tagging=object_tag)
        audit_logger.info('Successfully uploaded audit message to s3.')
    except Exception as err:
        msg = 'Failed to upload audit message to s3: {}'.format(err)
        audit_logger.warn(msg)
