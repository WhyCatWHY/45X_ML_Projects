import version as version
from setuptools import setup, find_packages

setup(
    name='MECH 45X PyTorch Image Classification',
    version=version.__version__,
    packages=find_packages(),
    url='https://github.com/ChenyiZheng/45X_ML_Projects.git',
    author='Chenyi Zheng // Henry Situ // Team 10 // MECH 45X',
    description='Python repo for MECH 45X machine learning tutorials',
    install_requires=[
        'pandas >= 1.1.4',
        'scikit-image',
        'numpy',
    ],
)