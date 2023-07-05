from setuptools import setup, find_packages
from pkg_resources import parse_requirements

with open('requirements.txt') as f:
    requirements = [str(req) for req in parse_requirements(f)]



setup(
    name='sts',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Shahar Zuler',
    author_email='shahar.zuler@gmail.com',
    description='A package for 3d shape correspondence based on https://github.com/omriefroni/STS',
    url='https://github.com/shaharzuler/STS',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)