from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="easyroutine",
    version="0.2",
    packages=find_packages(),
    description="A utility library for routine tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Francesco Ortu",
    author_email="francescortu@live.it",
    url="https://github.com/francescortu/easyroutine",
    install_requires=requirements,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)