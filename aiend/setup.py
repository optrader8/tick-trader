"""Setup script for tick-trader package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tick-trader",
    version="0.1.0",
    author="Tick Trader Team",
    description="선물옵션 틱데이터 기반 AI 트레이딩 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tick-trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pyarrow>=5.0.0",
        "redis>=4.0.0",
        "tensorflow>=2.10.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.5.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
    ],
)
