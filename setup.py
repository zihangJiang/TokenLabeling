from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'tlt',
  packages = find_packages(exclude=['seg','visualize']),
  version = '0.2.0',
  license='Apache License 2.0',
  long_description=long_description,
  long_description_content_type='text/markdown',
  description = 'Token Labeling Toolbox for training image models',
  author = 'Zihang Jiang',
  author_email = 'jzh0103@gmail.com',
  url = 'https://github.com/zihangJiang/TokenLabeling',
  keywords = [
    'imagenet',
    'attention mechanism',
    'transformer',
    'image classification',
    'token labeling'
  ],
  install_requires=[
    'timm>=0.4.5',
    'torch>=1.5',
    'torchvision',
    'scipy',
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
