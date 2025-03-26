from setuptools import find_packages, setup
import platform
import sys

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8', errors='ignore') as ff:
    required = ff.read().splitlines()

# if platform.system() == 'Darwin' and platform.processor() == 'arm':   # Specific for Apple M1 chips
#     required.append('scikit-learn')
#     required.append('statsmodels')
# else:
#     required.append('antspyx==0.4.2')

setup(
    name='translocdet',
    packages=find_packages(
        include=[
            'translocdet',
            'translocdet.Utils',
            'translocdet.Processing',
            'translocdet.Processing.Classical',
            'translocdet.Processing.AI',
            'tests',
        ]
    ),
    entry_points={
        'console_scripts': [
            'translocdet = translocdet.__main__:main'
        ]
    },
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.8",
    version='1.0.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Automatic detection of translocations on chromosome images',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
