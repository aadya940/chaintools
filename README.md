# chaintools - A Markov Chain Library in C

chaintools is a C library for working with Markov chains, providing functionalities to create, manipulate, and predict sequences using Markov models. This library is designed to be flexible, efficient, and easy to use.

## Features

- Create Markov chains with text-based states
- Build chains automatically from text files
- Set custom transition probability matrices
- Two prediction modes:
  - Deterministic prediction using `predict()`
  - Stochastic simulation using `simulate()`
- Efficient matrix operations using GNU Scientific Library (GSL)

## Installation on Linux

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

Include the library in your C program:
```c
#include "chaintools.h"

// Create a chain from states
char* states[] = {"A", "B", "C"};
markovchain chain = create_chain(states, 3);

// Or create from text file
markovchain* chain = create_chain_from_file("input.txt");

// Make predictions
char** prediction = predict(&chain, 5, "A");

// Or run simulations
char** simulation = simulate(&chain, 5, "A");
```

Compile your program as follows:
```bash
gcc -o your_program your_program.c -L/path/to/libchaintools -lchaintools -lgsl -lgslcblas -lm
```

### Installation of Python API

After compiling `chaintools` to `libchaintools.a`, head to the `python/pychaintools` where `setup.py` exists.

Execute as follows:
```bash
python setup.py build_ext --inplace
pip install .
```

Check installation by running `test.py` in `lib`:
```bash
python test.py
```
