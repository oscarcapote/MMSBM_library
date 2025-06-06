from setuptools import setup, find_packages

setup(
    name="BiMMSBM",
    version="0.1.0",
    packages=find_packages(include=['BiMMSBM', 'BiMMSBM.*']),
    package_dir={'BiMMSBM': '.'},
    install_requires=[
        "numpy",
        "numba",
        "scipy",
        "pandas",
        "matplotlib",
    ],
    author="Oscar Fajardo Fontiveros",
    author_email="your.email@example.com",  # Replace with your email
    description="A library for Mixed Membership Stochastic Block Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
) 