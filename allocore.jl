using JLD
using Distributions
using LinearAlgebra
using Random
using StatsFuns: logsumexp
using StatsBase
using Base.Threads
using Base.Filesystem

#helper functions 
include("utils.jl")

#multi-threading for training speedup
ENV["JULIA_NUM_THREADS"] = "4"

#function to fit an allocore decomposition, makes transition to python easier
function main(budget, latent, data, mask_bool=false, mask_diagonal=false, hp=0.01, burn=1000, iter=4000, save_each=20, dirichlet_rate = 0.0001, eps=1.0, zet=1.0, alph=1.0, bet=10.0, init=true)
    global tensor = data
    @assert sum(tensor) > 0 #check if tensor is not empty

    #hyperparameters
    global Q = budget
    global epsilon = eps
    global zeta = zet
    global alpha = alph
    global beta = bet

    #initialize model
    if (init == true)
        model = init_model(tensor, Q, alpha, beta, epsilon, zeta, latent)
        global holdout = mask_bool #masking to have heldout data or not
        global mask_diag = mask_diagonal #mask diagonal
        MCAR = false #missing completely at random
        p = hp #heldout probability
        global heldout, mask = gen_mask(MCAR, p, model["options"]["obs_dims"], mask_diag)
        model["mask"] = mask 
        model["to_impute"] = heldout
        hyperoptions = model["hyperoptions"]

        global options = model["options"]
        global M = length(latent) #number of modes

        @assert M == length(size(tensor))
        @assert M == length(latent)
        @assert Q <= prod(model["options"]["latent_dims"])

        global latent_dims = latent #latent dimensions
        options["latent_dims"] = latent
        model["options"]["latent_dims"] = latent
        global obs_dims = model["options"]["obs_dims"]
        @assert length(latent_dims)==length(obs_dims)
        global test_indices, diag_indices = train_test_split(tensor, heldout, mask, obs_dims) #split tensor into train and test sets

        global hyperoptions = model["hyperoptions"]
        hyperoptions["dirichlet_rate"] = dirichlet_rate
        global options = model["options"]
        global M = options["M"]
        @assert (size(heldout) == size(tensor)) & (size(mask) == size(tensor))
        @assert M == length(latent_dims)
        global b_llks = zeros(Float64, length(test_indices))
        global b_rates = copy(b_llks)
        global sample_dir = pwd()*"/samples/$Q"
        mkpath(sample_dir)
        @assert M == length(latent_dims)
    end



    #training details
    global burn_in = burn #number of burn-in samples
    global n_iter = iter #number of samples after burn-in
    global save_every = save_each #save every 'save_every' samples


    @assert (size(heldout) == size(tensor)) & (size(mask) == size(tensor))
    global b_llks = zeros(Float64, length(test_indices)) #for heldout likelihood computations
    global b_rates = copy(b_llks) 
    global sample_dir = pwd()*"/samples/$Q" #directory to save samples
    mkpath(sample_dir)

    println("specified core tensor size: $(latent_dims)")
    
    if (init == true) #initialize model
        initialize(hyperoptions, options, holdout)
    end

    posterior_samples = train_model(burn_in, n_iter, save_each, holdout) #train model, return samples
    return(posterior_samples)
end




 

