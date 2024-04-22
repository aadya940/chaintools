#include "../include/utils.h"
#include "../include/chaintools.h"

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#define INITIAL_CAPACITY 100
#define MAX_WORD_LENGTH 100

int __get_index(markovchain* chain, char* state){
    int idx = 0;
    
    for(int i = 0; i < chain->num_states; i++){
        char* __state = chain->states[i];
        if (strcmp(__state, state) == 0){
            return idx;
        }
        idx++;
    }
    
    return -1;
}

gsl_matrix* __array_to_gsl_matrix(float** matrix, size_t nrows, size_t ncols){
    gsl_matrix* transition_gslmat = gsl_matrix_alloc(nrows, ncols);
    for(size_t i = 0; i < nrows; i++){
        for(size_t j = 0; j < ncols; j++) {
            gsl_matrix_set(transition_gslmat, i, j, (double)matrix[i][j]);
        }
    }
    return transition_gslmat;
}

size_t __argmax_gsl(const gsl_vector* v){
    double max_val = gsl_vector_get(v, 0);
    size_t max_index = 0;

    for (size_t i = 1; i < v->size; i++) {
        double val = gsl_vector_get(v, i);
        if (val > max_val) {
            max_val = val;
            max_index = i;
        }
    }

    return max_index;
}

gsl_matrix* __build_vector_from_state(markovchain* chain, int state){
        /*Initial-Vector*/
    gsl_matrix* initial_vector = gsl_matrix_alloc(1, chain->num_states);
    gsl_matrix_set_zero(initial_vector);
    gsl_matrix_set(initial_vector, 0, state, 1);
    return initial_vector;
}


void __check_transition_matrix(int num_states, float transition_matrix[][num_states], markovchain* chain){
    if (chain->num_states != num_states) {
        fprintf(stderr, "Wrong dimensions of the transition matrix");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < num_states; i++){
        float sum = 0.0;
        for(int j = 0; j < num_states; j++){
            sum = sum + transition_matrix[i][j];
        }

        if (!(fabs(sum - 1.0) <= 0.002)){
            fprintf(stderr, "Row Sum of the transition matrix must equal 1");
            exit(EXIT_FAILURE);
        }
    }
}

int __is_present(char** words, int count, char* word) {
    for (int i = 0; i < count; i++) {
        if (strcmp(word, words[i]) == 0){
            return 1;
        }
    }

    return 0;
}
