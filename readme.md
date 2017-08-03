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
make, g++, boost-python, cmake, python 2 (python 3 has been tested with compilation fail), MSVC2015 (for windows)
```

#### Requirements Python
```
numpy, psutil
```

#### Requirements Python Testing
```sh
matplotlib
```

#### Installation (Linux)

```bash
cd denseinferencewrapper
make all
sudo pip install .
```

#### CMake Installation (Tested on Windows with MSVC2015)
```
1. Create a build folder anywhere
2. CMake with specifying source and build folder
3. Configure (Python, Boost, Numpy)
4. Generate
5. build 
	- Windows
		1. Open .sln file in build folder
		2. Build Release version
		3. Copy <build-folder>/build/denseinference/lib/refine_3d/Release/dense_inference.dll (.so for linux) to <source-folder>/denseinference/lib
		4. Rename the extension from .dll/.so to .pyd
	- Linux
		(To be updated...)
8. Open CMD/Terminal at <source-folder>
9. pip install . (Permission error will occur if .sln file is open)
10. Copy boost_python-vc140-mt-1_64.dll (.so for linux) at <boost-dir>/lib64-msvc-14.0 to <python-dir>/Lib/site-packages/denseinference/lib
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