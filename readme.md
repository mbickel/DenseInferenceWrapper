# Dense Inference Wrapper

## Overview
An add-on to Krähenbühls DenseCRF for 3d grayscale image volumes and a corresponding python wrapper to handle numpy ndarrays.

It was designed and tested to sharpen soft-labeled classifier outputs for medical 3d image data, such as CT-volumes.

The wrapper makes it very easy to run Krähenbühls super fast DenseCRF on your classifier output (i.e. the output of your fully convolutional neural network) just by a single python method call.


This code was used in MICCAI 2016 paper ([arXiv link](https://arxiv.org/abs/1610.02177)) titled : 

```
Automatic Liver and Lesion Segmentation in CT Using Cascaded Fully Convolutional 
Neural Networks and 3D Conditional Random Fields
```

### Citation ###

If you have used this code in your research please use the following BibTeX for citation :
```
@Inbook{Christ2016,
title="Automatic Liver and Lesion Segmentation in CT Using Cascaded Fully Convolutional Neural Networks and 3D Conditional Random Fields",
author="Christ, Patrick Ferdinand and Elshaer, Mohamed Ezzeldin A. and Ettlinger, Florian and Tatavarty, Sunil and Bickel, Marc and Bilic, Patrick and Rempfler, Markus and Armbruster, Marco and Hofmann, Felix and D'Anastasi, Melvin and Sommer, Wieland H. and Ahmadi, Seyed-Ahmad and Menze, Bjoern H.",
editor="Ourselin, Sebastien and Joskowicz, Leo and Sabuncu, Mert R. and Unal, Gozde and Wells, William",
bookTitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="415--423",
isbn="978-3-319-46723-8",
doi="10.1007/978-3-319-46723-8_48",
url="http://dx.doi.org/10.1007/978-3-319-46723-8_48"
}
```
You can find the full code at [https://github.com/IBBM/Cascaded-FCN](https://github.com/IBBM/Cascaded-FCN). 

## DenseCRF ``denseinference/lib``
The CRF-code is modified from the publicly available code by Philipp Krähenbühl and Vladlen Koltun.
See their project [website](http://www.philkr.net/home/densecrf) for more information.

If you also use this code, please cite their [paper](http://googledrive.com/host/0B6qziMs8hVGieFg0UzE0WmZaOW8/papers/densecrf.pdf):
Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, Philipp Krähenbühl and Vladlen Koltun, NIPS 2011.

The modification is also inspired by DeepLab and its application of DenseCRF. See their project [website](https://bitbucket.org/deeplab/deeplab-public)
for more information.

## Setup

#### Requirements

```
make, g++, boost-python
```

#### Requirements Python
```
numpy, psutil
```

#### Requirements Python Testing
```sh
matplotlib
```

#### Installation

```bash
cd denseinferencewrapper
make all
sudo pip install .
```

## Usage

```python
from denseinference import CRFProcessor

# init wrapper object
# all crf options can be set here (optional)
pro = CRFProcessor.CRF3DProcessor()
```

**param max_iterations**: maximum crf iterations
**param pos_x_std**: DenseCRF Param (float) (3)
**param pos_y_std**: DenseCRF Param (float) (3)
**param pos_z_std**: DenseCRF Param (float) (3)
**param pos_w**: DenseCRF Param (float) (3)
**param bilateral_x_std**: DenseCRF Param (float) (60)
**param bilateral_y_std**: DenseCRF Param (float) (60)
**param bilateral_z_std**: DenseCRF Param (float) (60)
**param bilateral_intensity_std**: DenseCRF Param (float) (20)
**param bilateral_w**: DenseCRF Param (float) (10)
**param dynamic_z**: Auto adjust z-params by image shape (bool) (False)
**param ignore_memory**: If true, images requiring to much memory are skipped (bool) (False)
**param verbose**: Print lot's of status information (bool) (False)

```python
# Now run crf and get hard labeled result tensor:
result = pro.set_data_and_run(img, feature_tensor)
```

**param img**: Normalized input as ndarray. (W, H, D), [0, 1]
**param label**: Continuous label tensor as ndarray. (W, H, D, L), [0, 1]
**return**: Hard labeled result as ndarray. (W, H, D), [0, L], dtype=int16
