from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fedtime",
    version="1.0.0",
    author="Raed Abdel-Sater, A. Ben Hamza",
    author_email="hamza@ciise.concordia.ca",
    description="A Federated Large Language Model for Long-Term Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FedTime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fedtime-train=run_longExp:main",
            "fedtime-federated=run_federated:main",
        ],
    },
    keywords="federated learning, time series forecasting, large language models, LLM, transformer, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/FedTime/issues",
        "Source": "https://github.com/yourusername/FedTime",
        "Documentation": "https://github.com/yourusername/FedTime#readme",
        "Paper": "https://arxiv.org/abs/2407.20503",
    },
)