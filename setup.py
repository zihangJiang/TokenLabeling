from setuptools import setup, find_packages

setup(
  name = 'tlt',
  packages = find_packages(exclude=['seg','visualize']),
  version = '0.1.0',
  license='Apache License 2.0',
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