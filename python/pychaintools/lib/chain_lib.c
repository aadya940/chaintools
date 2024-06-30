#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "../include/chaintools.h"
#include <numpy/arrayobject.h>

static PyObject* _create_chain_from_file(PyObject* self, PyObject* args) {
    char* fname;

    if (!PyArg_ParseTuple(args, "s", &fname)) {
        return NULL;
    }

    markovchain* chain;
    chain = create_chain_from_file(fname);

    if (chain == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Markov Chain from file");
        return NULL;
    }

    int rows = chain->num_states;
    int cols = chain->num_states;
    npy_intp dims[2] = {rows, cols};

    float** tpm = malloc(chain->num_states * chain->num_states * sizeof(float));
    memcpy(tpm, chain->tpm, chain->num_states * chain->num_states * sizeof(float));

    PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, (void*)tpm);

    if (arr == NULL) {
        PyErr_Print();
        fprintf(stderr, "Error creating NumPy array\n");
        return;
    }

    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);

    return arr;
}

static PyMethodDef ChainMethods[] = {
    {"_create_chain_from_file", _create_chain_from_file, METH_VARARGS, "Create a Markov chain from a file and return its transition probability matrix as a NumPy array"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef _libmodule = {
    PyModuleDef_HEAD_INIT,
    "chain_lib",          // Module name
    NULL,            // Module documentation (could be NULL)
    -1,              // Size of per-interpreter state of the module or -1 if the module keeps state in global variables
    ChainMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_chain_lib(void) {
    import_array();  // Initialize numpy array API
    return PyModule_Create(&_libmodule);
}
