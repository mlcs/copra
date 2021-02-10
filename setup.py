from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

version = {}
with open(os.path.join(_here, 'copra', 'version.py')) as f:
    exec(f.read(), version)

setup(
    name='copra',
    version=version['__version__'],
    description=('Construction of Proxy Record Age Models'),
    long_description=long_description,
    author='Bedartha Goswami',
    author_email='bedartha.goswami@uni-tuebingen.de',
    url='https://github.com/mlcs/copra',
    license='GNU Affero GPL v3.0',
    packages=['copra'],
#   no dependencies in this example
    install_requires=[
          'numpy==1.19.2',
          'pandas==1.1.3',
          'scipy==1.5.3',
         ],
#   no scripts in this example
#   scripts=['bin/a-script'],
    include_package_data=True,
    classifiers=[ 
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Programming Language :: Python :: Only',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        ]
    )
