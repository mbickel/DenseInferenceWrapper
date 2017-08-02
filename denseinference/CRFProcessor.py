__author__ = 'Marc Bickel, modified by Jacky Ko'

import numpy as np
import psutil
from ctypes import *
from sys import platform

if platform == "linux" or platform == "linux2":
  di_lib = 'lib/dense_inference.so'
elif platform == "win32":
  di_lib = 'lib/dense_inference.dll'

di = cdll.LoadLibrary(di_lib)

# from denseinference.lib import dense_inference as di

class CRF3DProcessor(object):
    # #
    #
    # Constructor
    #
    # #
    def __init__(self,
                 max_iterations=10,
                 pos_x_std=3.0,
                 pos_y_std=3.0,
                 pos_z_std=3.0,
                 pos_w=3.0,
                 bilateral_x_std=60.0,
                 bilateral_y_std=60.0,
                 bilateral_z_std=60.0,
                 bilateral_intensity_std=20.0,
                 bilateral_w=10.0,
                 dynamic_z=False,
                 ignore_memory=False,
                 verbose=False):
        """
        Initializes DenseCRF by Kraehenbuehl to refine volume labeling.
        Features are voxel distance and voxel intensity.

        Can be tuned by adjusting the optional parameters of the constructor.

        Expected data input:    Image tensor and label tensor.
                                Both image and label voxels must be normalized to [0, 1]
                                Image: (W, H, D); Label (W, H, D, C)

        Result:                 Result tensor.
                                Result tensor is hard labeled (W, H, D)

        :param max_iterations: maximum crf iterations
        :param pos_x_std: DenseCRF Param (float) (3)
        :param pos_y_std: DenseCRF Param (float) (3)
        :param pos_z_std: DenseCRF Param (float) (3)
        :param pos_w: DenseCRF Param (float) (3)
        :param bilateral_x_std: DenseCRF Param (float) (60)
        :param bilateral_y_std: DenseCRF Param (float) (60)
        :param bilateral_z_std: DenseCRF Param (float) (60)
        :param bilateral_intensity_std: DenseCRF Param (float) (20)
        :param bilateral_w: DenseCRF Param (float) (10)
        :param dynamic_z: Auto adjust z-params by image shape (bool) (False)
        :param ignore_memory: If true, images requiring to much memory are skipped (bool) (False)
        :param verbose: Print lot's of status information (bool) (False)
        """

        # private vars
        self.__verbose = verbose
        self.__ignore_mem = ignore_memory

        # adaptive z
        self.__dynamic_z = dynamic_z
        self.__bilateral_z_std = bilateral_z_std

        # init processor
        self.__processor = di.DenseCRF3DProcessor(max_iterations,
                                                  pos_x_std,
                                                  pos_y_std,
                                                  pos_z_std,
                                                  pos_w,
                                                  bilateral_x_std,
                                                  bilateral_y_std,
                                                  bilateral_z_std,
                                                  bilateral_intensity_std,
                                                  bilateral_w,
                                                  verbose)

    # #
    #
    # Static memory helpers
    #
    # #
    @staticmethod
    def memory_usage(label_shape):
        """
        Get anticipated memory requirements for labels_shape.
        :param label_shape: 4-dimensional label shape as tuple
        :return: required memory in bytes
        """
        if len(label_shape) != 4:
            raise ValueError('Error. 4-dimensional tensor expected. Got: ' + str(len(label_shape)))

        return di.DenseCRF3DProcessor.get_memory_requirements(label_shape[0],
                                                              label_shape[1],
                                                              label_shape[2],
                                                              label_shape[3])

    @staticmethod
    def memory_free():
        """
        Get current free system memory.
        :return: free memory in bytes
        """
        return psutil.virtual_memory().free

    @staticmethod
    def memory_percent(label_shape):
        """
        Get required memory for label_shape as fraction of free virtual memory as percent.
        :param label_shape: 4-dimensional label shape as tuple
        :return: required memory of free memory in percent
        """
        return float(CRF3DProcessor.memory_usage(label_shape)) / float(CRF3DProcessor.memory_free()) * 100.0

    # #
    #
    # Single exposed method to run CRF
    #
    # #
    def set_data_and_run(self, img, label):
        """
        Pass raw image volume and label tensor of your classifier (4d), run the CRF and receive refined hard labeled
        output. All input data and the returned output are ndarrays.
        :param img: Normalized input as ndarray. (W, H, D), [0, 1]
        :param label: Continuous label tensor as ndarray. (W, H, D, L), [0, 1]
        :return: Hard labeled result as ndarray. (W, H, D), [0, L], dtype=int16
        """
        if (type(img).__module__ != np.__name__) or (type(label).__module__ != np.__name__):
            raise ValueError('Error. Ndarray expected. Image: (W, H, D), [0, 1]; Label: (W, H, D, L), [0, 1].')

        if img.ndim != 3:
            raise ValueError('Error. 3d tensor expected. Got: ' + str(img.ndim))

        if label.ndim != 4:
            raise ValueError('Error. 4d tensor expected. Got: ' + str(label.ndim))

        # check image to label shape consistency
        if (img.shape[0] != label.shape[0]) or (img.shape[1] != label.shape[1]) or (img.shape[2] != label.shape[2]):
            raise ValueError('Error. Image shape and label shape inconsistent: ' +
                             str(img.shape) + ', ' + str(label.shape))

        # set image
        self.__check_and_set_img(img)

        # set label
        self.__check_and_set_label(label)

        # run crf
        res_arr = self.__run_crf(label.shape)

        return CRF3DProcessor.__prepare_and_return_result(res_arr, img.shape)

    # #
    #
    # Private convenience methods
    #
    # #
    def __check_and_set_img(self, img):
        if img.ndim != 3:
            raise ValueError('Error. 3d tensor expected. Got: ' + str(img.ndim))

        if np.amin(img) < 0 or np.amax(img) > 1:
            raise ValueError('Error. Image must be normalized to [0, 1].')

        # pass image to crf
        self.__processor.set_image(np.array(img, dtype=np.float64),
                                   img.shape[0],
                                   img.shape[1],
                                   img.shape[2],
                                   'float64')

        return img.shape

    def __check_and_set_label(self, label):
        if label.ndim != 4:
            raise ValueError('Error. 4d tensor expected. Got: ' + str(label.ndim))

        if np.amin(label) < 0 or np.amax(label) > 1:
            raise ValueError('Error. Label must be normalized to [0, 1].')

        # pass labels to crf
        self.__processor.set_feature(np.array(label, dtype=np.float64),
                                     label.shape[0],
                                     label.shape[1],
                                     label.shape[2],
                                     label.shape[3],
                                     'float64')

        return label.shape

    @staticmethod
    def __prepare_and_return_result(res_arr, shape):
        if len(shape) != 3:
            raise ValueError('Error. 3-d shape expected. Got: ' + str(len(shape)))

        if len(res_arr) != shape[0] * shape[1] * shape[2]:
            raise ValueError('Error. Length of array inconsisten with length of shape (' +
                             str(len(res_arr)) + ', ' + str(shape[0] * shape[1] * shape[2]) + ').')

        # Reshape with fortran order
        return np.asarray(res_arr, dtype=np.int16).reshape(shape, order='F')

    # #
    #
    # CRF
    #
    # #
    def __run_crf(self, label_shape):
        # check memory
        if (not self.__ignore_mem) and (self.memory_percent(label_shape) > 100.0):
            raise RuntimeError('Current image exceeds available memory. Free vs Needed: (' +
                               str(round(self.memory_free()/10**9, 2)) + 'GB, ' +
                               str(round(self.memory_usage(label_shape)/10**9, 2)) + 'GB).')

        if self.__dynamic_z:
            xy_mean = np.mean([label_shape[0], label_shape[1]])
            if (xy_mean != label_shape[2]) and (xy_mean != 0):
                bi_z_std = self.__bilateral_z_std * (float(label_shape[2]) / xy_mean)
                if self.__verbose:
                    print('Set Bilateral Z Std to ' + str(bi_z_std) + '.')
                self.__processor.set_bi_z_std(bi_z_std)

        return self.__processor.calc_res_array()


if __name__ == '__main__':
    print('All modules loaded successfully.')
