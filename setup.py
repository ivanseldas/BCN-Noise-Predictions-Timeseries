# setup.py
from setuptools import setup, find_packages

setup(
    name="bcn_noise",
    version="0.1",
    packages=find_packages(include=["src", "pipelines"]),
    package_dir={
        "src": "src",
        "pipelines": "pipelines"
    },
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ]
)