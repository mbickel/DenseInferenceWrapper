# Dense Inference Wrapper

## Overview
An add-on to Krähenbühls DenseCRF for 3d grayscale image volumes and a corresponding python wrapper to handle numpy ndarrays.

It was designed and tested to sharpen soft-labeled classifier outputs for medical 3d image data, such as CT-volumes.

The wrapper makes it very easy to run Krähenbühls super fast DenseCRF on your classifier output (i.e. the output of your fully convolutional neural network) just by a single python method call.

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

- **param max_iterations**: maximum crf iterations  
- **param pos_x_std**: DenseCRF Param (float) (3)  
- **param pos_y_std**: DenseCRF Param (float) (3)  
- **param pos_z_std**: DenseCRF Param (float) (3)  
- **param pos_w**: DenseCRF Param (float) (3)  
- **param bilateral_x_std**: DenseCRF Param (float) (60)  
- **param bilateral_y_std**: DenseCRF Param (float) (60)  
- **param bilateral_z_std**: DenseCRF Param (float) (60)  
- **param bilateral_intensity_std**: DenseCRF Param (float) (20)  
- **param bilateral_w**: DenseCRF Param (float) (10)  
- **param dynamic_z**: Auto adjust z-params by image shape (bool) (False)  
- **param ignore_memory**: If true, images requiring to much memory are skipped (bool) (False)  
- **param verbose**: Print lot's of status information (bool) (False)  

```python
# Now run crf and get hard labeled result tensor:
result = pro.set_data_and_run(img, feature_tensor)
```

- **param img**: Normalized input as ndarray. (W, H, D), [0, 1]  
- **param label**: Continuous label tensor as ndarray. (W, H, D, L), [0, 1]  
- **return**: Hard labeled result as ndarray. (W, H, D), [0, L], dtype=int16


# Porting to Python3
If you want to use this in `python3` then you can follow the following steps

#### Requirements

```
make, g++, boost-python3.5
```

### Installation of Boost for Python3


#####Install python 3.5 :
```sudo apt-get install python3-dev
```

### Make sure you have the libboost-python-dev libraries installed:

```sudo apt-get install libboost-python-dev
```

#####Download the lastest version of Boost from http://www.boost.org/

#####Run bootstrap with correct flags :
```./bootstrap.sh --with-python=python3.5
```

#####Compile Boost in directory :
```./b2
```

#####Update the Makefile to take the new boost locations (Already did)
 
