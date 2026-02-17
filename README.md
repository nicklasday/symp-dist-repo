# Symplectic Distribution

A Python package for working with symplectic distributions and Cartan geometry.

## Installation

To install the package, clone the repository and run:

```bash
pip install -e .
```

To install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Project Structure

```
symp-dist-repo/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── src/
│   └── symp_dist/          # Main package
│       ├── __init__.py
│       ├── algebra/        # Algebraic structures
│       ├── cartan_geometries/  # Cartan geometry module
│       ├── distributions/  # Distribution-related classes
│       └── utils/          # Utility functions
└── tests/                  # Test suite
```

## Running Tests

To run the test suite:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src/symp_dist tests/
```

## Features

- **Algebra Module**: Handles various algebraic structures including:
  - Cochain complexes
  - Tanaka symbols
  - Tensor algebras
  - Exterior algebras

- **Cartan Geometries**: Support for Cartan geometry constructions

- **Distributions**: Distribution classes for differential geometry

- **Utilities**: Helper functions and mathematical utilities

## Development

This project uses:
- `pytest` for testing
- `mypy` for static type checking
- `sympy` for symbolic mathematics

## License

[Add your license here]

## Author

Nicklas Day
