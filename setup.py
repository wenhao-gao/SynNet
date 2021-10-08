import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synth_net",
    version="1.0.0",
    description="Synth Net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wenhao-gao/synth_net/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "synth_net"},
    packages=setuptools.find_packages(where="synth_net"),
    python_requires=">=3.6",
)
