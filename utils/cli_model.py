"""
Runner for a CLI based model that reads file input and writes file output.

"""

import subprocess
import shutil
import os
from timeit import default_timer as timer
import logging
logger = logging.getLogger('cli_model')
import pydicom

class CliModel():
    """
    Runs a CLI based model taking a list of binary dicom parts and returning a list of dicom
    parts as output.

    Note: reads all output files in memory before returning, if memory is limited consider changing
    implementation to stream the output using file pointers.
    """

    def __init__(self, input_path, output_path, cleanup_paths, cli_args, cwd):
        """
        Instantiates a CliModel runner with arguments:
        - input_path: the directory the model will read input files from,
          where input dicom_parts will be written
        - output_path: the directory the model will write output files to
        - cleanup_paths: a list of paths that are created and removed after each call to 'run'
        - cli_args: a list of cli args to invoke the model (subprocess.run args)
        - cwd: the path to invoke the cli command from (subprocess cwd)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.cleanup_paths = cleanup_paths
        self.cli_args = cli_args
        self.cwd = cwd

    def run(self, request_binary_dicom_parts):
        timings = []

        try:
            start_time = timer()
            self._make_dirs()
            timings.append('make_dirs [{0:.2f}s]'.format(timer() - start_time))

            start_time = timer()
            self._write_model_input(request_binary_dicom_parts)
            timings.append('write_model_input {} parts [{:.2f}s]'.format(len(request_binary_dicom_parts), timer() - start_time))

            start_time = timer()
            self._invoke_cli_model()
            timings.append('run_cli_model [{0:.2f}s]'.format(timer() - start_time))

            start_time = timer()
            response_json_body, response_files = self._read_model_output()
            timings.append('read_model_output [{0:.2f}s]'.format(timer() - start_time))
        finally:
            start_time = timer()
            self._cleanup_dirs()
            timings.append('cleanup_dirs [{0:.2f}s]'.format(timer() - start_time))

        logger.debug('Timings:\n{}'.format('\n'.join(timings)))
        return response_json_body, response_files


    def _make_dirs(self):
        for path in self.cleanup_paths:
            os.makedirs(path, exist_ok=True)

    def _cleanup_dirs(self):
        for path in self.cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)


    def _write_model_input(self, binary_dicom_parts):
        for i, part in enumerate(binary_dicom_parts):
            file_path = os.path.join(self.input_path, "part{}.dcm".format(i))
            with open(file_path, "wb") as f:
                f.write(part.getbuffer())


    def _invoke_cli_model(self):
        subprocess.run(self.cli_args, cwd=self.cwd, text=True)

    def _read_model_output(self):
        # [ (dirname, subdirs, filenames) ]
        walk_dirs = [x for x in os.walk(self.output_path) if x[0] != '.']

        parts = []
        response_files = []

        i = 0
        for walk_dir in walk_dirs:
            dirname, _, filenames = walk_dir
            for filename in filenames:
                i += 1
                read_path = os.path.join(dirname, filename)
                ds = pydicom.dcmread(read_path, stop_before_pixels=True, specific_tags=['SeriesInstanceUID', 'SOPInstanceUID'])
                response_files.append(open(read_path, 'rb').read())

                parts.append({
                    'label': 'sc-{}'.format(i),
                    'binary_type': 'dicom_secondary_capture',
                    'SeriesInstanceUID': ds.SeriesInstanceUID,
                    'SOPInstanceUID': ds.SOPInstanceUID
                })


        response_json = {
            'protocol_version': '1.0',
            'parts': parts
        }

        return response_json, response_files
