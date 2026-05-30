from setuptools import setup, find_packages

setup(
    name="ecg-arrhythmia-detector",
    version="1.0.0",
    author="Braden Francis",
    description="ECG Signal Analysis and Arrhythmia Detection System using MIT-BIH Database",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "wfdb>=4.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "jinja2>=3.1.0",
        "click>=8.1.0",
    ],
)
