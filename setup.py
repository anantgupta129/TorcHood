from setuptools import find_packages, setup

VERSION = "0.2.0"
DESCRIPTION = "A Pytorch & lightning  Wrapper, for rapid prototyping"

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="torchood",
    version=VERSION,
    author="Anant Gupta",
    author_email="anantgupta129@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anantgupta129/TorcHood",
    packages=find_packages(exclude=["*tutorials*", "*notebooks*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
