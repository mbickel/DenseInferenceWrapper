from setuptools import setup, find_packages
from sys import platform

package_data = 'lib/dense_inference.pyd'

setup(
    name="DenseInferenceWrapper",
    version="1.0.0",
    packages=find_packages(),
    package_data={'': [package_data]},
    include_package_data=True,
    url='',
    license='',
    author='Marc Bickel, Jacky Ko',
    author_email='marc.bickel@mytum.de, jackkykokoko@gmail.com',
    description='Wrapper for Kraehenbuehls DenseCRF for 3D image data.'
)