from setuptools import setup, find_packages

setup(
    name='lyner',
    version='0.4.1',
    packages=find_packages(),
    url='',
    license='',
    author='Till Hartmann',
    author_email='till.hartmann@udo.edu',
    description='',
    install_requires=["plotly>=4.3",
                      #"plotly-orca>=1.2",
                      "psutil>=5.6",
                      "numpy>=1.17",
                      "numba>=0.46",
                      "networkx>=2.4",
                      "tensorflow>=2.0",
                      "keras>=2.3",
                      "pandas>=0.25",
                      "pybedtools>=0.8",
                      "scipy>=1.3",
                      "click>=7.0",
                      "pymc3>=3.7",
                      "scikit-learn>=0.21",
                      "mlxtend>=0.17",
                      "joblib>=0.14",
                      "networkx>=2.4",
                      "natsort>=6.2",
                      "click-aliases",
                      "Cluster_Ensembles>=1.16"
                      ],
    entry_points={'console_scripts': ["lyner= lyner.main:main"]}
)
