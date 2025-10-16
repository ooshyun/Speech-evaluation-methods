from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="speech-evaluation-methods",
    version="0.1.0",
    author="Seunghyun Oh",
    description="Evaluation Metrics for Speech Enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Speech-evaluation-methods",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Audio/Speech Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "soundfile>=0.10.0",
        "librosa>=0.9.0",
        "museval>=0.4.0",
        "pystoi>=0.3.0",
        "pesq>=0.0.4",
        "torch>=1.11.0",
        "torchaudio>=0.11.0",
        "torchmetrics>=0.8.0",
        "resampy>=0.2.0",
        "auraloss",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "pytest>=7.0.0",
        ],
    },
)

