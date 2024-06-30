import numpy as np
from chaintools import build_markovchain_from_file

chain = build_markovchain_from_file("../../../test.txt")

assert chain.tpm is not None
assert chain.states is not None
assert chain.num_states is not None
assert isinstance(chain.tpm, np.ndarray)
