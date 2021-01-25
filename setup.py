import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="akid",
    version="0.2",
    author="Shuai Li",
    author_email="lishuai918@gmail.com",
    description="A python package to study the theory of Deep Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shawnLeeZX/akid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
