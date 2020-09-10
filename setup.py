import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    'numpy',
    'torch==1.6.*',
    'torchvision==0.7.*',
    'Pillow==7.*',
    'pytorch-lightning==0.9.*',
    'umap-learn==0.4.*',
    'fvcore',
    'psutil',
    'matplotlib==3.2.*',
    'tensorboard==2.2.*',
    'future==0.18.*',
]


setup(
    name="ddbg",
    version="0.0.2",
    url="https://github.com/ml-illustrated/ddbg",
    author="ML Illustrated",
    author_email="gerald@ml-illustrated.com",
    description="Debugger for ML Datasets",
    #long_description=read("README.md"),
    long_description="Debugger for ML Datasets",
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6"
    ],
)
