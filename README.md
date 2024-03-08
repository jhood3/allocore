# Code for Al $\ell_0$ core
John Hood and Aaron Schein, University of Chicago

Source code for "The Al $\ell_0$ core Tensor Decomposition for Sparse Count Data", presented at *AISTATS 2024*. 

## What's included in src:

* [allocore.py](src/allocore.py): Python interface for running julia implementation of al $\ell_0$ core. 
* [allocore.jl](src/allocore.jl): Juli al $\ell_0$ core implementation.
* [utils.jl](src/utils.jl): Sampling functions and support for `allocore.jl`. 
* [allocore_tutorial.ipynb](src/allocore_tutorial.ipynb): Demo fitting an all $\ell_0$ core model to toy data in python.


## Dependencies:
Julia:
* [JLD](https://github.com/JuliaIO/JLD.jl)
* [Distributions](https://github.com/JuliaStats/Distributions.jl)
* [LinearAlgebra](https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/LinearAlgebra.jl)
* [Random](https://github.com/JuliaLang/julia/blob/master/stdlib/Random/docs/src/index.md)
* [StatsFuns](https://github.com/JuliaStats/StatsFuns.jl)
* [StatsBase](https://github.com/JuliaStats/StatsBase.jl)

Python:
* [numpy](https://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [pyjulia](https://github.com/JuliaPy/pyjulia)
