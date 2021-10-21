import setuptools

setuptools.setup(
    name="WIBAM",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=["torch>=1.6"],
)