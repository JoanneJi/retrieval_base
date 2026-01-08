# Installation Guide

## Quick Installation

### Method 1: Install using setup.py (Recommended)

```bash
cd src
pip install -e .
```

This will install the package in editable mode, allowing your code changes to take effect immediately.

### Method 2: Install dependencies only

```bash
cd src
pip install -r requirements.txt
```

Then you need to manually add the `src/` directory to Python path, or run using `python -m` method.

## Dependencies

### Standard Dependencies
- **numpy**: Numerical computing
- **scipy**: Scientific computing tools
- **pandas**: Data processing
- **astropy**: Astronomy library
- **PyAstronomy**: Astronomy utilities
- **matplotlib**: Plotting library
- **corner**: Corner plot tool

### Special Dependencies (May require additional configuration)

#### 1. petitRADTRANS
Atmospheric radiative transfer model, may need to be installed from source or a specific version.

Installation methods:
```bash
# Method 1: Install from PyPI (if available)
pip install petitRADTRANS

# Method 2: Install from source
git clone https://github.com/MarkusBonse/petitRADTRANS.git
cd petitRADTRANS
pip install -e .
```

#### 2. pymultinest
Nested sampling library, requires the MultiNest C library to be installed first.

Installation steps:
1. Install MultiNest C library (see: https://johannesbuchner.github.io/PyMultiNest/install.html)
2. Install Python bindings:
```bash
pip install pymultinest
```

## Verify Installation

After installation, you can verify by running:

```bash
cd src
python -c "from retrieval.retrieval import Retrieval; print('Installation successful!')"
```

## Run Example

After installation, you can run the example script:

```bash
cd src
python simple_retrieval.py
```

Or if installed using entry_points:

```bash
retrieval-run
```

## Development Mode Installation

If you need to develop or modify the code:

```bash
cd src
pip install -e ".[dev]"
```

This will install development dependencies (e.g., pytest, black, flake8, etc.).

## Troubleshooting

### Issue 1: Module not found
**Solution**: Make sure to run from the `src/` directory, or use the `python -m` method:
```bash
python -m src.simple_retrieval
```

### Issue 2: pymultinest installation fails
**Solution**: Make sure the MultiNest C library is installed and the correct environment variables are set.

### Issue 3: petitRADTRANS import error
**Solution**: Check that the pRT input data path is correctly set, refer to the configuration in `core/paths.py`.

