#ifndef UTILS_H
#define UTILS_H

#include "chaintools.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

typedef struct {
    char** words;
    int size;
    int capacity;
} UniqueWordsArray;


void __check_transition_matrix(int num_states, float transition_matrix[][num_states], markovchain* chain);
int __get_index(markovchain* chain, char* state);
gsl_matrix* __array_to_gsl_matrix(float** matrix, size_t nrows, size_t ncols);
gsl_matrix* __build_vector_from_state(markovchain* chain, int state);
size_t __argmax_gsl(const gsl_vector* v);
int __is_present(char** words, int count, char* word);

#endif /* UTILS_H */