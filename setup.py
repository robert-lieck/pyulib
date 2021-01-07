import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyulib",
    version="0.1.1",
    author="Robert Lieck",
    author_email="robert.lieck@epfl.ch",
    description="python utility functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robert-lieck/pyulib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
