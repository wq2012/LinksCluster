"""Setup script for the package."""

import setuptools

VERSION = "0.1.0"

with open("README.md", "r") as file_object:
  LONG_DESCRIPTION = file_object.read()

with open("requirements.txt") as file_object:
  INSTALL_REQUIRES = file_object.read().splitlines()

setuptools.setup(
    name="linkscluster",
    version=VERSION,
    author="Quan Wang",
    author_email="quanw@google.com",
    description="Links: A High-Dimensional Online Clustering Method",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/LinksCluster",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
)
