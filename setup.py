from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hydroopt",  
    version="0.1.0",         
    author="Gladistony Silva Lins",
    description="Biblioteca de otimização de redes hidráulicas (EPANET) utilizando algoritmos de Inteligência de Enxame (GWO, WOA, PSO) para minimização de custos e garantia de pressão.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "wntr>=1.0.0",
        "mealpy>=2.5.0",
        "openpyxl",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "gpu-cuda": ["torch>=1.9.0"],  # Para detecção e uso de GPU CUDA
        "gpu-cupy": ["cupy>=9.0.0"],   # Alternativa de GPU com CuPy
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"],  # Desenvolvimento
    },
) 