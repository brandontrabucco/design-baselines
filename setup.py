from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['tensorflow==2.2',
                     'matplotlib']


PACKAGES = [package
            for package in find_packages() if
            package.startswith('design_baselines')]


setup(name='design_baselines',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=PACKAGES,
      description='Baselines for Model-Based Optimization')
