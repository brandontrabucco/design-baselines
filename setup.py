from setuptools import find_packages
from setuptools import setup


PACKAGES = [
    'tensorflow==2.2', 'tensorflow-probability',
    'gym[mujoco]', 'numpy', 'pandas', 'matplotlib',
    'seaborn', 'click', 'ray[tune]']


setup(
    name='forward-model',
    description='Forward model optimization with hyperparameter tuning',
    license='MIT',
    version='0.1',
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    install_requires=PACKAGES,
    entry_points={
        'console_scripts': ('forward-model=forward_model.main:cli',)})
