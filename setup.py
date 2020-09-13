with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zotbin",
    version="0.0.1",
    author="David Kirkby",
    author_email="dkirkby@uci.edu",
    description="Tomographic binning for cosmology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dkirkby/zotbin",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'matplotlib', 'jax', 'jax-cosmo', 'jax-flows'],
)
