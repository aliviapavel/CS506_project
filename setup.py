from setuptools import setup, find_packages

setup(
    name="adipocyte_annotator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'scanpy>=1.9.1',
        'scikit-learn>=1.0.2',
        'pandas>=1.3.5',
        'numpy>=1.21.6',
        'scipy>=1.7.3',
        'matplotlib>=3.5.3',
        'seaborn>=0.11.2',
        'joblib>=1.1.0',
    ],
)