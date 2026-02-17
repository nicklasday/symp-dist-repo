# Symplectified Distributions

A Python package for working with the symplectification of rank 2 vector distributions and their Cartan geometries, as constructed in the 2009 paper "On local geometry of non‐holonomic rank 2 distributions" of B. Doubrov and I. Zelenko, or as outlined in the 2025 paper "Symplectification of Rank 2 Distributions, Normal Cartan Connections, and Cartan Prolongations" of N. Day, B. Doubrov, and I. Zelenko (you can find an ArXiv version of this paper [here](https://arxiv.org/abs/2506.09232).

Note that this is a working research tool, not a published library.

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
