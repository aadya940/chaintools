#ifndef CHAITOOLS_H
#define CHAITOOLS_H

/**
 * @brief Structure representing a Markov chain
 * 
 * Contains the transition probability matrix (tpm), states, and number of states
 * for a Markov chain.
 */
typedef struct {
    int num_states;      /**< Number of states in the Markov chain */
    float** tpm;         /**< Transition probability matrix */
    char** states;       /**< Array of state labels */
} markovchain;

/**
 * @brief Creates a new Markov chain with given states
 * 
 * @param chain_states Array of strings representing state labels
 * @param num_states Number of states in the chain
 * @return markovchain Initialized Markov chain structure
 */
markovchain create_chain(char** chain_states, int num_states);

/**
 * @brief Sets the transition probability matrix for a Markov chain
 * 
 * @param num_states Number of states in the chain
 * @param transition_matrix 2D array containing transition probabilities
 * @param chain Pointer to the Markov chain to modify
 */
void set_transition_matrix(int num_states, float transition_matrix[][num_states], markovchain* chain);

/**
 * @brief Predicts the most likely sequence of states
 * 
 * @param chain Pointer to the Markov chain
 * @param n_steps Number of steps to predict
 * @param initial_state Starting state
 * @return char** Array of predicted states
 */
char** predict(markovchain* chain, int n_steps, char* initial_state);

/**
 * @brief Creates a Markov chain from a text file
 * 
 * @param fname Path to the input text file
 * @return markovchain* Pointer to the created Markov chain
 */
markovchain* create_chain_from_file(char* fname);

/**
 * @brief Simulates a random walk through the Markov chain
 * 
 * @param chain Pointer to the Markov chain
 * @param n_steps Number of steps to simulate
 * @param initial_state Starting state
 * @return char** Array of simulated states
 */
char** simulate(markovchain* chain, int n_steps, char* initial_state);

#endif /* CHAINOPY_H */