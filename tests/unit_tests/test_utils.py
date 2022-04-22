import unittest

import pydicom
from PIL import Image
from tempfile import TemporaryDirectory

from arterys_sdk.inference_test_tool.utils import convert_image_to_dicom, load_image_data, sort_series_images, \
    filter_masks_by_binary_type


class TestUtils(unittest.TestCase):
    def test_convert_image_to_dicom(self):
        with TemporaryDirectory() as tmp_dir:
            img = Image.new("RGB", (25, 25), (255, 255, 255))
            img_path = f'{tmp_dir}/test_img.jpg'
            img.save(img_path)
            dcm_path = convert_image_to_dicom(img_path)
            dcm_file = pydicom.dcmread(dcm_path)
            self.assertIsInstance(dcm_file, pydicom.dataset.FileDataset)

    def test_sort_series_images(self):
        test_images = load_image_data('./tests/data/test_3d/')
        self.assertEqual(len(test_images), 3)
        sorted_images = sort_series_images(test_images)

        # images should be in order of 3->2->1
        self.assertEqual(sorted_images[0].position, test_images[-1].position)
        self.assertEqual(sorted_images[-1].position, test_images[0].position)

        # change position which should cause images to be in order of 1->2->3
        test_images[0].position[-1] = -30
        test_images[-1].position[-1] = 100
        sorted_images = sort_series_images(test_images)
        self.assertEqual(sorted_images[0].position, test_images[0].position)
        self.assertEqual(sorted_images[-1].position, test_images[-1].position)

        # mock multi timepoint series
        multi_timepoint_test_images = [test_images[0], test_images[0], test_images[1]]
        with self.assertRaises(AssertionError):
            sort_series_images(multi_timepoint_test_images)

        # Test missing position/direction returns unsorted images
        test_images = load_image_data('./tests/data/test_3d/')
        test_images[0].position = None
        sorted_images = sort_series_images(test_images)
        self.assertEqual(sorted_images[0].position, test_images[0].position)

    def test_filter_masks_by_binary_type(self):
        """ Test filtering of json response """

        # check whether empty masks works
        empty_masks = []
        empty_response = {
            'protocol_version': '1.0',
            'parts': [],
        }

        returned_masks, returned_sc = filter_masks_by_binary_type(empty_masks, empty_response)
        self.assertTrue(returned_masks.size == 0)
        self.assertTrue(returned_sc.size == 0)


if __name__ == '__main__':
    unittest.main()
