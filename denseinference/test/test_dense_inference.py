__author__ = 'mbickel'

import unittest
import numpy as np

from denseinference.lib import dense_inference as di
import caffehelper.nifti_helper as nh

import matplotlib.pyplot as plt

class DenseTestClass(unittest.TestCase):

    def setUp(self):
        self.processor = di.DenseCRF3DProcessor()

    def test_processor_returns_error_message_if_image_or_features_not_set(self):
        self.assertRaises(RuntimeError, self.processor.calc_res_array)

    def test_processor_returns_error_message_if_image_not_float64(self):
        self.assertRaises(Exception, self.processor.set_image, np.ones((512, 512, 600), np.float64), 512, 512, 600, 'float63')

    def test_processsor_takes_numpy_image(self):
        self.processor.set_image(np.ones((100, 100, 50), np.float64), 100, 100, 50, 'float64')
        # fac 255
        self.assertEqual(np.multiply(np.array(self.processor.get_image()).size, 255),
                         np.sum(np.array(self.processor.get_image())))

    def test_processor_takes_numpy_features(self):
        self.processor.set_feature(np.ones((100, 100, 50, 3), np.float64), 100, 100, 50, 3, 'float64')
        self.assertEqual(np.array(self.processor.get_feature()).size,
                         np.sum(np.array(self.processor.get_feature())))

    def test_processor_image_position(self):
        arr = np.zeros((2, 2, 2))
        arr[1, 1, 0] = 1
        arr[1, 1, 1] = 1
        arr[1, 0, 0] = 1
        self.processor.set_image(np.array(arr, np.float64), arr.shape[0], arr.shape[1], arr.shape[2], 'float64')

        res = np.array(self.processor.get_image())

        # Important here is the order option 'F' for serialization (fortran order), which is used internally in wrapper
        self.assertEqual(np.ravel(np.multiply(arr, 255), order='F').tostring(), res.tostring())

    def test_processor_feature_position(self):
        arr = np.zeros((2, 2, 2, 2))
        arr[1, 1, 0, 1] = 1

        self.processor.set_feature(np.array(arr, np.float64),
                                   arr.shape[0],
                                   arr.shape[1],
                                   arr.shape[2],
                                   arr.shape[3],
                                   'float64')
        res = np.array(self.processor.get_feature())

    #@unittest.SkipTest
    def test_processor_processing_ones_is_0_and_keeps_size(self):
        # set image
        self.processor.set_image(np.ones((50, 50, 50), np.float64), 50, 50, 50, 'float64')

        # set features
        self.processor.set_feature(np.ones((50, 50, 50, 50), np.float64), 50, 50, 50, 50, 'float64')

        # process and get results
        res_arr = np.asarray(self.processor.calc_res_array(), dtype=np.int16)
        self.assertEqual(np.sum(res_arr), 0)
        self.assertEqual(res_arr.size, 125000)

    #@unittest.SkipTest
    def test_processor_processing_random_not_equal(self):

        # first result
        self.processor.set_image(np.array(np.random.rand(100, 100, 100), np.float64), 100, 100, 100, 'float64')
        self.processor.set_feature(np.array(np.random.rand(100, 100, 100, 3), np.float64), 100, 100, 100, 3, 'float64')
        res_arr1 = np.asarray(self.processor.calc_res_array(), dtype=np.int16)

        # second result
        self.processor.set_image(np.array(np.random.rand(100, 100, 100), np.float64), 100, 100, 100, 'float64')
        self.processor.set_feature(np.array(np.random.rand(100, 100, 100, 3), np.float64), 100, 100, 100, 3, 'float64')
        res_arr2 = np.asarray(self.processor.calc_res_array(), dtype=np.int16)

        # third result
        self.processor.set_image(np.array(np.random.rand(100, 100, 100), np.float64), 100, 100, 100, 'float64')
        self.processor.set_feature(np.array(np.random.rand(100, 100, 100, 3), np.float64), 100, 100, 100, 3, 'float64')
        res_arr3 = np.asarray(self.processor.calc_res_array(), dtype=np.int16)

        self.assertFalse(np.array_equal(res_arr1, res_arr2) and np.array_equal(res_arr1, res_arr3))

    #@unittest.SkipTest
    def test_processor_processing_test_image_2label_and_keep_image(self):
        # create image
        img = np.zeros((40, 40, 40), dtype=np.float64)
        img_flat = np.ravel(img)

        for i, el in enumerate(img_flat):
            if i < len(img_flat)/2:
                img_flat[i] = 0
            else:
                img_flat[i] = 1

        img = img_flat.reshape(40, 40, 40)

        # create label
        label_a = np.copy(img)
        label_b = np.copy(img)

        # inverse label_b
        label_b = np.add(np.multiply(label_b, -1), 1)

        # combine label_a with label_b
        seg_tensor = np.concatenate(([label_b], [label_a]), axis=0)
        seg_tensor = np.swapaxes(seg_tensor, 0, 3)
        seg_tensor = np.swapaxes(seg_tensor, 0, 1)
        seg_tensor = np.swapaxes(seg_tensor, 1, 2)

        # set image
        self.processor.set_image(img, 40, 40, 40, 'float64')

        # set features
        self.processor.set_feature(seg_tensor, 40, 40, 40, 2, 'float64')

        # process and get results
        res_arr = np.asarray(self.processor.calc_res_array(), dtype=np.int16)

        res_vol = res_arr.reshape(40, 40, 40, order='F')
        self.assertTrue(np.array_equal(img, res_vol))

    #@unittest.SkipTest
    def test_processor_processing_test_image_3label_and_keep_image(self):
        # create image
        img = np.zeros((60, 60, 60), dtype=np.float64)
        img_flat = np.ravel(img)

        for i, el in enumerate(img_flat):
            if i < len(img_flat)/3:
                img_flat[i] = 0
            elif i < len(img_flat)*(2.0/3.0) and i >= len(img_flat)/3:
                img_flat[i] = 1
            else:
                img_flat[i] = 2

        img = img_flat.reshape(60, 60, 60)

        seg_tensor = np.zeros(img.shape+(3,))

        # hack labels
        for label in xrange(3):
            mask = img == label
            seg_tensor[mask, label] = 1

        self.assertNotEqual(np.sum(seg_tensor), 60*60*60*3)
        self.assertEqual(np.sum(seg_tensor[..., 0]), 60*60*20)
        self.assertEqual(np.sum(seg_tensor[..., 1]), 60*60*20)
        self.assertEqual(np.sum(seg_tensor[..., 2]), 60*60*20)

        # scale image down
        img = np.multiply(img, 0.5)

        # set image
        self.processor.set_image(img, 60, 60, 60, 'float64')

        # set features
        self.processor.set_feature(seg_tensor, 60, 60, 60, 3, 'float64')

        # process and get results
        res_arr = np.asarray(self.processor.calc_res_array(), dtype=np.int16)

        res_vol = res_arr.reshape(60, 60, 60, order='F')

        self.assertTrue(np.array_equal(np.multiply(img, 2), res_vol))

if __name__ == '__main__':
    unittest.main()