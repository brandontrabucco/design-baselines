from setuptools import find_packages
from setuptools import setup


setup(
    name='forward-model',
    description='Forward model optimization with hyperparameter tuning',
    license='MIT',
    version='0.1',
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    entry_points={
        'console_scripts': ('forward-model=forward_model.cli:cli',)})
