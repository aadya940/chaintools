# chaintools - A Markov Chain Library in C

chaintools is a C library for working with Markov chains, providing functionalities to create, manipulate, and predict sequences using Markov models. This library is designed to be flexible, efficient, and easy to use.

## Features

- Create Markov chains from scratch or from input files.
- Set transition matrices for the Markov chains.
- Predict next states in the chain based on initial states.
- Simple and intuitive interface.
- Extend the Python Interpreter for Fast Algorithms

## Installation on linux

Clone the repository:

```bash
git clone https://github.com/yourusername/chaintools.git
```

Install GNU Scientific Library: 

```bash
sudo apt-get install libgsl-dev
```

Use `make` to compile the library:
```bash
make
```

### Usage

```
#include "chaintools.h"
```

Compile the file as follows:
```
gcc -o your_program your_program.c -L/path/to/libchaintools -lchaintools -lgsl -lgslcblas -lm
```

### Installation of Python API

After compiling `chaintools` to `libchaintools.a`, head to the `python/pychaintools` where `setup.py` exists.

- Execute as follows:
```bash
python setup.py build_ext --inplace
pip install .
```

- Check Installation by run `test.py` in `lib`
```
python test.py
```
