#ifndef CHAINOPY_H
#define CHAINOPY_H

typedef struct {
    int num_states;
    float** tpm;
    char** states;
} markovchain;


markovchain create_chain(char** chain_states, int num_states);
void set_transition_matrix(int num_states, float transition_matrix[][num_states], markovchain* chain);
char** predict(markovchain* chain, int n_steps, char* initial_state);
markovchain* create_chain_from_file(char* fname);


#endif /* CHAINOPY_H */