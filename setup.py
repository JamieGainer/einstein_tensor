import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="einstein_tensor",
    version="0.0.1",
    author="Jamie Gainer",
    author_email="jgainer137@gmail.com",
    description="A lightweight implementation of tensors which follow the Einstein summation convention when multiplied",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JamieGainer/einstein_tensor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)