from setuptools import setup, find_packages

setup(
    name="MMSBM_library",
    version="0.1.0",
    packages=find_packages(include=['MMSBM_library', 'MMSBM_library.*']),
    package_dir={'MMSBM_library': '.'},
    install_requires=[
        "numpy",
        "numba",
        "scipy",
        "pandas",
        "matplotlib",
    ],
    author="Oscar Fajardo Fontiveros",
    author_email="oscarcapote@hotmail.es",  # Replace with your email
    description="A library for Mixed Membership Stochastic Block Models in bipartite networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
) 