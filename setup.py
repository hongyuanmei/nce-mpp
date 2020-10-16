import setuptools
from setuptools import setup

setup(name='nce_point_process',
      version='0.1',
      description='nce for point processeses',
      packages=setuptools.find_packages(),
      zip_safe=False,
      install_requires=[
          'torch==1.1.0',
          'numpy', 'tqdm', 'psutil', 'matplotlib'
      ],
      test_suite='nose.collector',
      tests_requires=['nose'])
