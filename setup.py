from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hydroopt",  
    version="0.0.1",         
    author="Gladistony Silva Lins",
    description="Pacote de teste de publicação",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[], 
)