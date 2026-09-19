"""
QuantFinance-EDU Package Setup
"""
from setuptools import setup, find_packages

setup(
    name="quantfin_edu",
    version="1.0.0",
    description="Seven-Module Quantitative Financial Analysis System (Educational)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.36",
        "scikit-learn>=1.3.0",
        "cvxpy>=1.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "statsmodels>=0.14.0",
        "requests>=2.28.0",
    ],
)
