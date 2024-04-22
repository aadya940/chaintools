#include "../include/chaintools.h"
#include "../include/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <ctype.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>


/*
Markov Chain
------------

SETUP
-----
create_chain()
create_chain_from_file()
set_transition_matrix()

FUNCTIONS
---------
predict()
*/

#define MAX_UNIQUE_WORD_LIMIT 800

markovchain create_chain(char** chain_states, int num_states) {
    float** tpm = (float**)malloc(num_states * sizeof(float*));

    if(tpm == NULL){
        fprintf(stderr, "Filed to allocate memory for the transition-matrix");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < num_states; i++){
        tpm[i] = (float*)malloc(num_states * sizeof(float));
        if(tpm[i] == NULL){
        fprintf(stderr, "Filed to allocate memory for the transition-matrix");
        exit(EXIT_FAILURE);
        }
    }

    markovchain chain;
    chain.tpm = tpm;
    chain.states = chain_states;
    chain.num_states = num_states;

    return chain;
}


void set_transition_matrix(int num_states, float transition_matrix[][num_states], markovchain* chain){
    __check_transition_matrix(num_states, transition_matrix, chain);

    for(int i = 0; i < num_states; i++){
        memcpy(chain->tpm[i], transition_matrix[i], num_states * sizeof(float));
    }

    return;
}


char** predict(markovchain* chain, int n_steps, char* initial_state){
    char** _sim_results = (char**)malloc(n_steps * sizeof(char*));
    int* _temp_results = (int*)malloc(n_steps * sizeof(int));

    if (_sim_results == NULL) {
        fprintf(stderr, "Failed to allocate Memory for the Matrix.");
        exit(EXIT_FAILURE);
    }

    if (_temp_results == NULL) {
        fprintf(stderr, "Failed to allocate Memory for the Matrix.");
        exit(EXIT_FAILURE);
    }

    int __idx = __get_index(chain, initial_state);
    memcpy(_temp_results, &__idx, sizeof(int));

    gsl_matrix* tpm = __array_to_gsl_matrix(chain->tpm, chain->num_states, chain->num_states);
    gsl_matrix* initial_vector = __build_vector_from_state(chain, __idx);

    for(int i = 1; i < n_steps; i++){
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, initial_vector, tpm, 0, initial_vector);
        gsl_vector* vector = gsl_vector_alloc(chain->num_states);
        gsl_matrix_get_row(vector, initial_vector, 0);
        int __idx = __argmax_gsl(vector);
        memcpy(_temp_results, &__idx, sizeof(int));
        initial_vector = __build_vector_from_state(chain, __idx);
    }

    for(int i = 0; i < n_steps; i++){
        int j = _temp_results[i];
        char* value = chain->states[j];
        _sim_results[i] = (char*)malloc((strlen(value) + 1) * sizeof(char));
        strcpy(_sim_results[i], value);
    }
    
    free(_temp_results);
    return _sim_results;
}

markovchain* create_chain_from_file(char* fname) {
   FILE* file_pointer = fopen(fname, "r");

    if (file_pointer == NULL){
        fprintf(stderr, "Error Read the File.");
        exit(EXIT_FAILURE);
    }

   char __word[100];
   char** __unique_words_array;
    __unique_words_array = (char**)malloc(MAX_UNIQUE_WORD_LIMIT * sizeof(char*));

    int num_unique = 0;
   while (fscanf(file_pointer, "%s", __word) != EOF) {
    if (num_unique == 0) {
        __unique_words_array[0] = (char*)malloc((strlen(__word) + 1) * sizeof(char));
        strcpy(__unique_words_array[0], __word);
        num_unique++;
        continue;
    }

    if (num_unique == MAX_UNIQUE_WORD_LIMIT){
        __unique_words_array = (char**)realloc(__unique_words_array, MAX_UNIQUE_WORD_LIMIT * 2 * sizeof(char*));
    }

    int __isUnique = 0; 

    for (int i = 0; i < num_unique; i++){
        if (strcmp(__word, __unique_words_array[i]) != 0){
            __isUnique++;
        }

        if (__isUnique == num_unique) {
            __unique_words_array[num_unique] = (char*)malloc((strlen(__word) + 1) * sizeof(char));
            strcpy(__unique_words_array[num_unique], __word);
            num_unique++;
        }
    }    
    }
    
    rewind(file_pointer);

    int num_states = num_unique;    
    float transition_count[num_states][num_states];
    memset(transition_count, 0, sizeof(transition_count));

    markovchain* chain = (markovchain*)malloc(sizeof(chain));
    chain->num_states = num_states;
    chain->states = (char**)malloc(chain->num_states * sizeof(char*));

    for(int i = 0; i < num_unique; i++) {
        chain->states[i] = (char*)malloc((strlen(__unique_words_array[i]) + 1) * sizeof(char));
        strcpy(chain->states[i], __unique_words_array[i]);
    }

    char* word = (char*)malloc(sizeof(char) * 100);
    char* nextWord = (char*)malloc(sizeof(char) * 100);

    fscanf(file_pointer, "%s", word);
    
    if (file_pointer == NULL){
        fprintf(stderr, "Error opening the file.\n");
        exit(EXIT_FAILURE);
    }

    while (fscanf(file_pointer, "%s", nextWord) == 1) {
        int __idx_word = __get_index(chain, word);
        int __idx_nextWord = __get_index(chain, nextWord);

        transition_count[__idx_word][__idx_nextWord]++;
        strcpy(word, nextWord);
    }    

    for (int i = 0; i < chain->num_states; i++){
        double sum = 0.0;

        for (int j = 0; j < chain->num_states; j++) {
            sum = sum + transition_count[i][j];
        }

        for (int j = 0; j < chain->num_states; j++) {
            if (sum == 0) {
            sum = 0.00001;
                }
            transition_count[i][j] /= sum;
        }
    }

    chain->tpm = (float**)malloc(num_states * sizeof(float*));
    
    for (int i = 0; i < num_states; i++){
        chain->tpm[i] = (float*)malloc(num_states * sizeof(float));
        memcpy(chain->tpm[i], transition_count[i], sizeof(transition_count[i]));
    }

    fclose(file_pointer);
    
    for(int i = 0; i < num_unique; i++) {
        free(__unique_words_array[i]);
    }

    return chain;
};
