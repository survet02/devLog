from setuptools import setup, find_packages

setup(
    name='faceguessr',
    version='0.0.5',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'faceguessr': ['requirements.txt']},
    install_requires=[],
)
