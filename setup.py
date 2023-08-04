from setuptools import find_packages, setup

VERSION = "0.1"
DESCRIPTION = "A Pytorch Wrapper"

setup(
    name="torchood",
    version=VERSION,
    packages=find_packages(),
    author="[Anant Gupta]",
    author_email="anantgupta129@gmail.com",
    install_requires=[
        "torch==2.0.1",
        "torchvision==0.15.2",
        "lightning=2.0.6",
        "albumentations==1.3.1",
        "torch-lr-finder",
        "torchinfo",
        "grad-cam",
        "torchmetrics==0.8.0",
    ],
)
