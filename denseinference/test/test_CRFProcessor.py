__author__ = 'mbickel'

import numpy as np

import unittest
from denseinference import CRFProcessor as crf_pro


class DenseWrapperTestClass(unittest.TestCase):

    def test_CRFProcessor_extended_constructor(self):
        pro = crf_pro.CRF3DProcessor()
        self.assertIsNotNone(pro)

    def test_CRFProcessor_memory_raises_value_error(self):
        self.assertRaises(ValueError, crf_pro.CRF3DProcessor.memory_usage, (512, 512, 800))
        self.assertRaises(ValueError, crf_pro.CRF3DProcessor.memory_percent, (512, 512, 800))

    def test_CRFProcessor_raises_shape_errors(self):
        # instantiate processor
        pro = crf_pro.CRF3DProcessor()

        # shape inconsistency
        self.assertRaises(ValueError, pro.set_data_and_run, np.ones((20, 20, 20)), np.ones((20, 30, 20, 3)))

        # wrong image shape
        self.assertRaises(ValueError, pro.set_data_and_run, np.ones((20, 20)), np.ones((20, 20, 20, 3)))

        # wrong label shape
        self.assertRaises(ValueError, pro.set_data_and_run, np.ones((20, 20, 20)), np.ones((20, 20, 20)))

        # wrong image range
        self.assertRaises(ValueError, pro.set_data_and_run, np.ones((20, 20, 20)) * 1.5, np.ones((20, 20, 20, 2)))

        # wrong label range
        self.assertRaises(ValueError, pro.set_data_and_run, np.ones((20, 20, 20)), np.ones((20, 20, 20, 2)) * (-0.1))

        # wrong data type
        self.assertRaises(ValueError, pro.set_data_and_run, 'hallo', 'welt')
        self.assertRaises(ValueError, pro.set_data_and_run, [1], [1])

    def test_CRFProcessor_processing_test_image_2label_and_keep_image(self):
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

        # instantiate processor
        pro = crf_pro.CRF3DProcessor()

        # run
        res_vol = pro.set_data_and_run(img, seg_tensor)

        self.assertTrue(np.array_equal(img, res_vol))

    def test_CRFProcessor_processing_zero_to_zero_and_outputting_verbose(self):
        pro = crf_pro.CRF3DProcessor(verbose=True, dynamic_z=True)

        img = np.zeros((40, 50, 80), dtype=np.float64)
        label = np.zeros((40, 50, 80, 20), dtype=np.float64)

        res = pro.set_data_and_run(img, label)
        self.assertEqual(np.sum(res), 0)

if __name__ == '__main__':
    unittest.main()
