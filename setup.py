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

        'coms=design_baselines.coms_cleaned:coms_cleaned',

        'coms-cleaned=design_baselines.'
        'coms_cleaned.experiments:cli',
        'coms-original=design_baselines.'
        'coms_original.experiments:cli',

        'gradient-ascent=design_baselines.'
        'gradient_ascent.experiments:cli',
        'gradient-ascent-relabelled=design_baselines.'
        'gradient_ascent.relabel_experiments:cli',
        'gradient-ascent-ablate-distribution=design_baselines.'
        'gradient_ascent.distribution_experiments:cli',

        'gradient-ascent-min-ensemble=design_baselines.'
        'gradient_ascent.min_ensemble_experiments:cli',
        'gradient-ascent-min-ensemble-relabelled=design_baselines.'
        'gradient_ascent.relabel_min_ensemble_experiments:cli',
        'gradient-ascent-min-ensemble-ablate-distribution=design_baselines.'
        'gradient_ascent.distribution_min_ensemble_experiments:cli',

        'gradient-ascent-mean-ensemble=design_baselines.'
        'gradient_ascent.mean_ensemble_experiments:cli',
        'gradient-ascent-mean-ensemble-relabelled=design_baselines.'
        'gradient_ascent.relabel_mean_ensemble_experiments:cli',
        'gradient-ascent-mean-ensemble-ablate-distribution=design_baselines.'
        'gradient_ascent.distribution_mean_ensemble_experiments:cli',

        'mins=design_baselines.'
        'mins.experiments:cli',
        'mins-relabelled=design_baselines.'
        'mins.relabel_experiments:cli',
        'mins-ablate-distribution=design_baselines.'
        'mins.distribution_experiments:cli',

        'cbas=design_baselines.'
        'cbas.experiments:cli',
        'cbas-relabelled=design_baselines.'
        'cbas.relabel_experiments:cli',
        'cbas-ablate-distribution=design_baselines.'
        'cbas.distribution_experiments:cli',

        'autofocused-cbas=design_baselines.'
        'autofocused_cbas.experiments:cli',
        'autofocused-cbas-relabelled=design_baselines.'
        'autofocused_cbas.relabel_experiments:cli',
        'autofocused-cbas-ablate-distribution=design_baselines.'
        'autofocused_cbas.distribution_experiments:cli',

        'cma-es=design_baselines.'
        'cma_es.experiments:cli',
        'cma-es-relabelled=design_baselines.'
        'cma_es.relabel_experiments:cli',
        'cma-es-ablate-distribution=design_baselines.'
        'cma_es.distribution_experiments:cli',

        'bo-qei=design_baselines.'
        'bo_qei.experiments:cli',
        'bo-qei-relabelled=design_baselines.'
        'bo_qei.relabel_experiments:cli',
        'bo-qei-ablate-distribution=design_baselines.'
        'bo_qei.distribution_experiments:cli',

        'reinforce=design_baselines.'
        'reinforce.experiments:cli',
        'reinforce-relabelled=design_baselines.'
        'reinforce.relabel_experiments:cli',
        'reinforce-ablate-distribution=design_baselines.'
        'reinforce.distribution_experiments:cli',

        'online-reinforce=design_baselines.'
        'reinforce.online_experiments:cli',
    )})
