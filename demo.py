__author__ = 'Wendong Xu'

from denseinference import CRFProcessor
import nibabel as nib
import numpy as np


slice_shape = (512, 512)
crf2d = CRFProcessor.CRF2DProcessor()
crf3d = CRFProcessor.CRF3DProcessor()


def histeq_processor(img):
	"""Histogram equalization"""
	nbr_bins=256
	#get image histogram
	imhist,bins = np.histogram(img.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum() #cumulative distribution function
	cdf = 255 * cdf / cdf[-1] #normalize
	#use linear interpolation of cdf to find new pixel values
	original_shape = img.shape
	img = np.interp(img.flatten(),bins[:-1],cdf)
	img=img/255.0
	return img.reshape(original_shape)


def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
  """ Converts from hounsfield units to float64 image with range 0.0 to 1.0 """
  # calc min and max
  min, max = np.amin(arr), np.amax(arr)
  if min <= 0:
    arr = np.clip(arr, min * c_min, max * c_max)
    # right shift to zero
    arr = np.abs(min * c_min) + arr
  else:
    arr = np.clip(arr, min, max * c_max)
    # left shift to zero
    arr = arr - min
  # normalization
  norm_fac = np.amax(arr)
  if norm_fac != 0:
    norm = np.divide(
      np.multiply(arr, 255),
      np.amax(arr))
  else:  # don't divide through 0
    norm = np.multiply(arr, 255)

  norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
  return norm


def process_img_label(imgvol, segvol):
  """
  Process a given image volume and its label and return arrays as a new copy
  :param imgvol:
  :param label_vol:
  :return:
  """
  imgvol_downscaled = np.zeros((slice_shape[0], slice_shape[1], imgvol.shape[2]))
  segvol_downscaled = np.zeros((slice_shape[0], slice_shape[1], imgvol.shape[2]))
  imgvol[imgvol > 1200] = 0

  for i in range(imgvol.shape[2]):
    # Get the current slice, normalize and downscale
    slice = np.copy(imgvol[:, :, i])
    slice = norm_hounsfield_dyn(slice)
    slice = histeq_processor(slice)
    imgvol_downscaled[:, :, i] = slice
    # downscale the label slice for the crf
    segvol_downscaled[:, :, i] = segvol[:,:,i] # to_scale(segvol[:, :, i], slice_shape)

  return [imgvol_downscaled, segvol_downscaled]


def crf_2d_processing(imgvol, probvol):
  '''
  2DCRF: set_data_and_run(img, prob)
  img: [W, H], [0, 1]
  prob: [W, H, L], [0, 1]
  :param imgvol: [W, H, D]
  :param probvol: [W, H, D, L]
  :return:
  '''
  global crf2d
  assert imgvol.shape == probvol.shape, 'Invalid imgvol and probvol length'
  vol_size = imgvol.shape[2]
  new_probvol = np.zeros((probvol.shape[0], probvol.shape[1], probvol.shape[3], probvol.shape[2]))
  new_probvol = probvol.transpose((0, 1, 3, 2))

  hard_cls = []
  for i in range(0, vol_size):
    print('slice {} finished.'.format(i))
    temp_cls = crf2d.set_data_and_run(imgvol[..., i], new_probvol[..., i])
    hard_cls.append(temp_cls)
  return hard_cls


def crf_3d_processing(imgvol, probvol):
  '''
  3DCRF: set_data_and_run(imgvol, probvol)
  imgvol: [W, H, D], [0, 1]
  probvol: [W, H, D, L], [0, 1]

  :param imgvol: [W, H, D]
  :param probvol: [W, H, D, L]
  :return:
  '''
  global crf3d
  assert imgvol.shape == probvol.shape, 'Invalid imgvol and probvol length'
  return crf3d.set_data_and_run(imgvol, probvol)


def get_validate_files(imgvol, segvol, probvol):
  '''
  :param imgvol:
  :param segvol:
  :param probvol:
  :return:
  '''
  new_probvol = np.zeros((probvol.shape[0], probvol.shape[1], probvol.shape[3], probvol.shape[2]))
  new_probvol = probvol.transpose((0, 1, 3, 2))
  new_probvol = new_probvol.astype(np.float64)
  new_probvol[new_probvol == 0] = 1e-18

  imgvol, segvol = process_img_label(imgvol, segvol)
  return imgvol, segvol, new_probvol


if __name__ == '__main__':
  imgvol_path = './data/volume-0.nii'
  segvol_path = './data/segmentation-0.nii'
  probvol_path = './data/test-logits-0.npz'

  imgvol = nib.load(imgvol_path).get_data()
  segvol = nib.load(segvol_path).get_data()
  probvol = np.load(probvol_path)['logits']

  imgvol, segvol, probvol = get_validate_files(imgvol, segvol, probvol)
  print(imgvol.shape, segvol.shape, probvol.shape)
  seg2dcrf = crf_2d_processing(imgvol, probvol)
  seg3dcrf = crf_3d_processing(imgvol, probvol)
