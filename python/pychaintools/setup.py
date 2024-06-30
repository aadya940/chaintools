from setuptools import setup, Extension
import numpy as np

extension_mod = Extension(
    "chain_lib",
    sources=[
        "lib/chain_lib.c",
        "../../src/chaintools.c",
        "../../src/utils.c",
    ],                                                                                      # List of source files
    include_dirs=["../../src", "../../include", np.get_include(), "/usr/include/gsl"],      # Include directories
    libraries=["chaintools", "gsl", "gslcblas"],                                            # Link against libchaintools.a
    library_dirs=["../../", "/usr/lib"],                                                    # Directory containing libchaintools.a
    extra_compile_args=["-std=c11"],                                                        # Optional: specify C standard
)

setup(
    name="chain_lib",
    version="1.0",
    description="Python extension module for Markov chain operations",
    ext_modules=[extension_mod],
    install_requires=["numpy"],                                                             # Ensure numpy is installed
)
