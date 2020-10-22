import setuptools
from setuptools import setup

setup(name='ncempp',
      version='1.0',
      description='noise-contrastive estimation for multivariate point process',
      packages=setuptools.find_packages(),
      zip_safe=False,
      install_requires=[
          'torch==1.1.0',
          'numpy', 'tqdm', 'psutil', 'matplotlib'
      ],
      test_suite='nose.collector',
      tests_requires=['nose'])
