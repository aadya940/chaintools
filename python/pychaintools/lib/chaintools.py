import chain_lib


def build_matrix_from_file(fname):
    _arr = chain_lib._create_chain_from_file(fname)
    return _arr
