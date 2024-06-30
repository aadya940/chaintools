from . import chain_lib


def build_markovchain_from_file(path):
    """
    Creates MarkovChain from string file located at `path`.

    Parameters
    ----------
    path: str
        path of the text file.

    Returns
    -------
    MarkovChain:
        tpm: ndarray
            Transition Matrix
        num_states: int
            Number of States of the Chain.
        states: List[str]
            Name of the states a.k.a Unique Words in a File.
    """
    _chain = chain_lib._create_chain_from_file(path)
    return _chain
