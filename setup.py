from setuptools import find_packages
from setuptools import setup


setup(
    name='design-baselines',
    description='Baselines for Model-Based Optimization',
    license='MIT',
    version='0.1',
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    entry_points={'console_scripts': (
        'design-baselines=design_baselines.cli:cli',
        'csm=design_baselines.csm.experiments:cli',
        'online=design_baselines.online.experiments:cli',
        'gradient-ascent=design_baselines.gradient_ascent.experiments:cli',
        'gan=design_baselines.gan.experiments:cli',
        'mins=design_baselines.mins.experiments:cli',
        'cbas=design_baselines.cbas.experiments:cli',
        'cma-es=design_baselines.cma_es.experiments:cli',
        'bo-qei=design_baselines.bo_qei.experiments:cli',
        'reinforce=design_baselines.reinforce.experiments:cli',
        'autofocus=design_baselines.autofocus.experiments:cli'
    )})
