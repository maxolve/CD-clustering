# setup.py

from setuptools import setup, find_packages

setup(
    name='cdclustering',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'networkx',
        'python-louvain',
        'scikit-learn',
        'scipy',
        'ucimlrepo'
    ],
)
