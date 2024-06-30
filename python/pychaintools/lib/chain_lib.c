#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "../include/chaintools.h"
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <stdio.h>

typedef struct {
    PyObject_HEAD
    int num_states;
    PyArrayObject* tpm;
    PyObject* states;
} MarkovChain;

static void MarkovChain_dealloc(MarkovChain* self) {
    Py_XDECREF(self->tpm);  
    Py_XDECREF(self->states);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyMemberDef MarkovChain_members[] = {
    {"num_states", T_INT, offsetof(MarkovChain, num_states), 0, "number of states"},
    {"tpm", T_OBJECT_EX, offsetof(MarkovChain, tpm), 0, "transition probability matrix"},
    {"states", T_OBJECT_EX, offsetof(MarkovChain, states), 0, "states"},
    {NULL}  
};

static PyTypeObject MarkovChainType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "pychaintools.MarkovChain",            /* tp_name */
    sizeof(MarkovChain),                   /* tp_basicsize */
    0,                                     /* tp_itemsize */
    (destructor)MarkovChain_dealloc,       /* tp_dealloc */
    0,                                     /* tp_print */
    0,                                     /* tp_getattr */
    0,                                     /* tp_setattr */
    0,                                     /* tp_as_async */
    0,                                     /* tp_repr */
    0,                                     /* tp_as_number */
    0,                                     /* tp_as_sequence */
    0,                                     /* tp_as_mapping */
    0,                                     /* tp_hash  */
    0,                                     /* tp_call */
    0,                                     /* tp_str */
    0,                                     /* tp_getattro */
    0,                                     /* tp_setattro */
    0,                                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                    /* tp_flags */
    "MarkovChain objects",                 /* tp_doc */
    0,                                     /* tp_traverse */
    0,                                     /* tp_clear */
    0,                                     /* tp_richcompare */
    0,                                     /* tp_weaklistoffset */
    0,                                     /* tp_iter */
    0,                                     /* tp_iternext */
    0,                                     /* tp_methods */
    MarkovChain_members,                   /* tp_members */
    0,                                     /* tp_getset */
    0,                                     /* tp_base */
    0,                                     /* tp_dict */
    0,                                     /* tp_descr_get */
    0,                                     /* tp_descr_set */
    0,                                     /* tp_dictoffset */
    0,                                     /* tp_init */
    0,                                     /* tp_alloc */
    0,                                     /* tp_new */
};           

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
    PyObject* states_list = PyList_New(chain->num_states);

    MarkovChain* __chain = (MarkovChain*)MarkovChainType.tp_alloc(&MarkovChainType, 0);
    
    for (int i = 0; i < chain->num_states; i++) {
        PyObject* state_str = PyUnicode_FromString(chain->states[i]);
        if (state_str == NULL) {
            Py_DECREF(arr);
            Py_DECREF(states_list);
            PyErr_SetString(PyExc_RuntimeError, "Error converting state to Python string");
            return NULL;
        }
        PyList_SET_ITEM(states_list, i, state_str);
    }

    if (__chain == NULL) {
        Py_DECREF(arr);
        Py_DECREF(states_list);
        return NULL;
    }

    __chain->tpm = (PyArrayObject*)arr;
    __chain->num_states = chain->num_states;
    __chain->states = states_list;

    return (PyObject*)__chain;
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
    import_array();
    if (PyType_Ready(&MarkovChainType) < 0) {
        return NULL;
    }

    PyObject* m = PyModule_Create(&_libmodule);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&MarkovChainType);
    PyModule_AddObject(m, "MarkovChain", (PyObject*)&MarkovChainType);
    return m;
}
