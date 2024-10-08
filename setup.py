import os
from setuptools import setup, find_packages

# Get the absolute path to the directory where setup.py is located
this_directory = os.path.abspath(os.path.dirname(__file__))

# Load the long description from the README
with open('README.md') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name="crosci",
    version="0.1.0",
    author="Arthur-Ervin Avramiea, Marina Diachenko",
    author_email="a.e.avramiea@vu.nl, a.e.avramiea@vu.nl",
    description="Critical oscillations biomarkers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arthur-ervin/crosci",
    packages=find_packages(include=['crosci', 'crosci.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    project_urls={
        "Homepage": "https://github.com/arthur-ervin/crosci",
        "Issues": "https://github.com/arthur-ervin/crosci/issues",
    },
)
