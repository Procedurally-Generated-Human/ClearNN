from setuptools import find_packages, setup
setup(
    name='clearNN',
    packages=find_packages(include=['clearNN']),
    version='0.0.1',
    description='Crystal clear object oriented neural network library',
    author='Parsa Toopchinezhad',
    license='MIT',
    install_requires=['numpy==1.24.2'],
)