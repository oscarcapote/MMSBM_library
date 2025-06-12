from setuptools import setup, find_packages

setup(
    name="BiMMSBM",
    version="0.1",
    packages=find_packages(include=['BiMMSBM', 'BiMMSBM.*']),
    package_dir={'BiMMSBM': '.'},
    install_requires=[
        'numpy',
        'pandas',
        'numba',
    ],
    author="Oscar Fajardo Fontiveros",
    author_email="oscarcapote@hotmail.es",  # Replace with your email
    description="A library for Mixed Membership Stochastic Block Models in bipartite networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
) 