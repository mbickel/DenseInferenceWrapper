from setuptools import setup, find_packages
setup(
    name="DenseInferenceWrapper",
    version="1.0.0",
    packages=find_packages(),
    package_data={'': ['lib/dense_inference.so']},
    include_package_data=True,
    url='',
    license='',
    author='Marc Bickel',
    author_email='marc.bickel@mytum.de',
    description='Wrapper for Kraehenbuehls DenseCRF for 3D image data.'
)