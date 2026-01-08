"""
Setup script for the exoplanet atmosphere retrieval framework.

This package provides a lightweight retrieval pipeline based on
petitRADTRANS (pRT3) and PyMultiNest, designed for high-resolution
emission spectroscopy.

Installation:
    pip install -e .

Or install dependencies only:
    pip install -r requirements.txt
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file if it exists
readme_file = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read version from __init__ or set default
version = "0.1.0"

setup(
    name="retrieval-base",
    version=version,
    description="Simple exoplanet atmosphere retrieval framework for high-resolution emission spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chenyang Ji",
    author_email="",  # Add your email if needed
    url="",  # Add repository URL if available
    packages=find_packages(where=".", exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        # Core scientific computing
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        
        # Astronomy
        "astropy>=5.0.0",
        "PyAstronomy>=0.18.0",
        
        # Atmospheric modeling
        "petitRADTRANS>=3.0.0",  # Note: May need to install from source or specific repository
        
        # Bayesian inference
        "pymultinest>=2.11.0",  # Note: Requires MultiNest library to be installed separately
        
        # Plotting
        "matplotlib>=3.5.0",
        "corner>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            # Note: This requires the package to be installed in editable mode
            # Run: cd src && pip install -e .
            "retrieval-run=simple_retrieval:main",
        ],
    },
    py_modules=["simple_retrieval"],  # Include simple_retrieval.py as a module
    include_package_data=True,
    zip_safe=False,
)

