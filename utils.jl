using JLD
using Distributions
using LinearAlgebra
using Random
using StatsFuns: logsumexp
using StatsBase
using Base.Threads
using Base.Filesystem


function init_factor(alpha::Float64, beta::Float64, obs_dims, latent_dims) # initialize factor matrices
    M = length(obs_dims)
    factor_matrices_M = Array{Matrix{Float64}}(undef, M)
    M = length(factor_matrices_M)
    for m in 1:M
        theta = rand(Gamma((sum(tensor[mask.==0])/Q)^(1/(M))/obs_dims[m], 1 / beta), obs_dims[m], latent_dims[m])
        factor_matrices_M[m] = theta
    end
    return(factor_matrices_M)
end

function init_core_rate(epsilon::Float64, zeta::Float64, Q::Int64) #initialize nonzero core values
    rate = rand(Gamma(epsilon, 1 / zeta), Q)
    return (rate)
end

function init_core_prior(latent_dims, obs_dims, prior = true) #initialize pi_M, prior on core index locations 
    M = length(latent_dims)
    pis_M = Array{Vector{Float64}}(undef, M)
    if prior == true
        for m in 1:M
            pis_M[m] = zeros(latent_dims[m])
            alpha = ones(latent_dims[m]).*0.0001/obs_dims[m]
            pis_M[m] = rand(Dirichlet(alpha))
        end
    else 
        for m in 1:M
            pis_M[m] = zeros(latent_dims[m])
            alpha = ones(latent_dims[m])./latent_dims[m]
            pis_M[m] = alpha
        end
    end
    return(pis_M)
end

function update_prior(latent_dims, pis_M, indices_QM, obs_dims, hyperoptions) #update prior on core index locations
    sharing_param = hyperoptions["dirichlet_rate"]
    M = length(pis_M)
    for m in 1:M
        alpha = fill(sharing_param/obs_dims[m], latent_dims[m])
        inds = indices_QM[:,m]
        alpha[inds] .+= 1
        pis_M[m] = rand(Dirichlet(alpha))
    end
    return(pis_M)
end


function init_core_indices(latent_dims, Q) #initialize core index locations
    M = length(latent_dims)
    indices_QM = zeros(Int64, Q, M)
        for m in 1:M
            for q in 1:Q
            indices_QM[q, m] = mod(q, latent_dims[m]) 
                if (indices_QM[q,m]==0)
                    indices_QM[q,m] = latent_dims[m]
                end
            end
        end
    return (indices_QM)
end


function init_allocate(obs_dims, Q) #initialize allocation matrices for allocation step
    M = length(obs_dims)
    count_matrices = Array{Matrix{Float64}}(undef, M)
    for i in 1:M
        matrix = zeros(obs_dims[i], Q)
        count_matrices[i] = matrix
    end
    return (count_matrices)
end


function allocate(nonzero_indices, nonzero_counts, factor_matrices_M, indices_QM, lambdas_Q, obs_dims) #allocate counts to latent classes
    Q = length(lambdas_Q)
    M = length(obs_dims)
    y_Q = zeros(Q)
    y_M = init_allocate(obs_dims, Q)
    nonzero_counts = Int.(round.(nonzero_counts))
    @assert length(nonzero_counts) == length(nonzero_indices)
    locker = Threads.SpinLock()
    @views @threads for j in eachindex(nonzero_indices)
        nz_ind = nonzero_indices[j]
        probs = ones(Q)
       @views for m in 1:M
            theta = factor_matrices_M[m]
            ind = nz_ind[m]
            cq = indices_QM[:, m]
            probs .*= theta[ind, cq]
            end
         probs.*=lambdas_Q
         probs2 = sum(probs) == 0 ? ones(Q)/Q : probs ./= sum(probs)
         counts = rand(Multinomial(nonzero_counts[j], probs2))[:,1]
        lock(locker)
        @views for m in 1:M
            ind = nz_ind[m]
            y_M[m][ind, :] += counts
        end 
        y_Q += counts
        unlock(locker)
    end
    @assert round(sum(y_Q)) == round(sum(nonzero_counts))
    @views for m in 1:M
        @assert round(sum(y_M[m])) == round(sum(nonzero_counts))
    end 
    return (y_M, y_Q)
end



function allocore(M, lambdas_Q, factor_matrices_M, indices_QM, y_M, pis_M, dirichlet=true, bound=1e9) #allocore step
    @views for m in 1:M
        theta1 = factor_matrices_M[m]
        latent = Int(size(theta1, 2))
        @views for q in 1:Q
            if dirichlet == true
                prior = pis_M[m]
            end
            constant = lambdas_Q[q]
            y_q = y_M[m][:, q]
            @views for j in 1:M
                if m != j
                    ind = indices_QM[q, j] #QxM
                    theta = factor_matrices_M[j]
                    constant *= sum(theta[:, ind])
                end
            end
            llk = -theta1 .* constant .+ y_q .* log.(theta1 .* constant) .+ log.(prior')
            llk[llk.<-bound] .= -bound
            llk[llk.>bound] .= bound
            llk[isnan.(llk)] .= -bound
            llks = dropdims(sum(llk, dims=1), dims=1)
            weights = exp.(llks .- logsumexp(llks))
            k_m_q = sample(1:latent, Weights(weights))
            indices_QM[q, m]  = k_m_q
        end
    end
    for m in 1:M
        unK = length(unique(indices_QM[:,m]))
    end
    return (indices_QM)
end




function update_factor(y_M, factor_matrices_M, latent_dims, obs_dims, lambdas_Q, alpha, beta, indices_QM, m) #resample factor matrices from complete conditional
    latent = latent_dims[m]
    obs = Int(obs_dims[m])
    gamma = zeros(latent)
    latent_count = zeros(obs, latent)
    y_i = y_M[m]
    for i in 1:latent
        inds = findall(==(i), indices_QM[:, m])
        latent_count[:, i] = Int.(sum(y_i[:, inds], dims=2))
        s = 0
        @views for q in inds
            temp = lambdas_Q[q]
            @views  for m0 in 1:M
                if m0 != m
                    theta = factor_matrices_M[m0]
                    temp *= sum(theta[:, indices_QM[q, m0]])
                end
            end
            s += temp
        end
        gamma[i] = s
    end
    @views for i in 1:obs
        @views for c in 1:latent
            factor_matrices_M[m][i,c] = rand(Gamma(alpha + latent_count[i, c], 1 / (beta + gamma[c])))
        end
    end
    return (factor_matrices_M)
end



function update_core_rate(y_Q, epsilon, zeta, indices_QM, factor_matrices_M, lambdas_Q) #resample nonzero core values from complete conditional
    M = length(factor_matrices_M)
    Q = length(y_Q)
    @views for q in 1:Q
        summation = 1
        @views for m in 1:M
            temp = factor_matrices_M[m][:, indices_QM[q, m]]
            @assert length(temp) == obs_dims[m]
            summation *= sum(temp)
        end
        proposal = rand(Gamma(epsilon + y_Q[q], 1 / (zeta + summation)), 1)[1]
        lambdas_Q[q] = proposal
    end
    return lambdas_Q
end



function impute(factor_matrices_M, test_indices, lambdas_Q, indices_QM, true_counts, barY, init=false)   #impute heldout data
    Q = length(lambdas_Q)
    M = length(factor_matrices_M)
    rates = zeros(length(test_indices))
    imputed = zeros(length(test_indices))
    likelihood = zeros(length(test_indices))
    bar_y = barY
    locker = SpinLock()
    if init == false
        @views @threads for i in eachindex(test_indices)
            temp = ones(Q)
            inds = test_indices[i]
            for j in 1:M
                theta = factor_matrices_M[j]
                temp .*= theta[inds[j], indices_QM[:,j]]
            end
            rate_q = temp.*lambdas_Q
            rate = sum(rate_q)
            imputed[i] = rand(Poisson(rate), 1)[1]
            rates[i] = rate
            likelihood[i] = pdf(Poisson(rate), true_counts[i])
        end 
    elseif init==true
        @views @threads for i in eachindex(test_indices)
            imputed[i] = rand(Poisson(bar_y), 1)[1]
            rates[i] = bar_y
            likelihood[i] = pdf(Poisson(bar_y), true_counts[i])
        end
    end  
    return(imputed, likelihood, rates)
end





function initialize(hyperparams, params, holdout) #initialize model
    obs = params["obs_dims"]
    latent = params["latent_dims"]
    Q = params["Q"]
    M = params["M"]
    alpha = hyperparams["alpha"]
    beta = hyperparams["beta"]
    epsilon = hyperparams["epsilon"]
    zeta = hyperparams["zeta"]
    global factor_matrices_M = init_factor(alpha, beta, obs, latent)
    global lambdas_Q = init_core_rate(epsilon, zeta, Q)
    global pis_M = init_core_prior(latent, obs, false)
    global indices_QM = init_core_indices(latent, Q)
    global nonzero_train_indices = findall(x -> x > 0, abs.(tensor.*(mask .- 1)))
    global barY = mean(tensor[mask.==0])
    global nonzero_train = tensor[nonzero_train_indices]
    global true_counts = tensor[test_indices]
    global diag_counts = tensor[diag_indices]
    if holdout == true
        global imputed, _, _ = impute(factor_matrices_M, test_indices, lambdas_Q, indices_QM, true_counts, barY, true)
    end
    if mask_diag == true
        global imputed_diag, _, _ = impute(factor_matrices_M, diag_indices, lambdas_Q, indices_QM,diag_counts, barY, true)
    end
    println("beginning training, Q = $(Q)")
end




function BPA(iter, split=true) #Bayesian Poisson Allocore transition kernel
    global y_M, y_Q = allocate_every_n(iter, 5)
    global indices_QM  = allocore(M, lambdas_Q, factor_matrices_M, indices_QM, y_M, pis_M, true)
    global pis_M = update_prior(latent_dims, pis_M, indices_QM, obs_dims, hyperoptions)
    global lambdas_Q = update_core_rate(y_Q, epsilon, zeta, indices_QM, factor_matrices_M, lambdas_Q)
    iteratively_update_factor()
    impute_missing(iter, split, mask_diag)
end

function BPCPTUCK(iter, split=true) #Bayesian Poisson CP, Bayesian Poisson Tucker transition kernels
    global y_M, y_Q = allocate_every_n(iter, 5)
    global lambdas_Q = update_core_rate(y_Q, epsilon, zeta, indices_QM, factor_matrices_M, lambdas_Q)
    iteratively_update_factor()
    impute_missing(iter, split, mask_diag)
end

function allocate_every_n(iter, n) #allocate counts to latent classes every n iterations
    if mod(iter, n)==0
        if (holdout==true) & (mask_diag == true)
            nzind_diag = findall(x -> x > 0, imputed_diag)
            nonzero_diag_indices = diag_indices[nzind_diag]
            nonzero_diag_counts = imputed_diag[nzind_diag]
            nzind = findall(x -> x > 0, imputed)
            nonzero_test_indices = test_indices[nzind]
            nonzero_test_counts = imputed[nzind]
            nonzero_indices = vcat(nonzero_train_indices, nonzero_test_indices, nonzero_diag_indices)
            nonzero_counts = vcat(nonzero_train, nonzero_test_counts, nonzero_diag_counts)
        elseif holdout==true
            nzind = findall(x -> x > 0, imputed)
            nonzero_test_indices = test_indices[nzind]
            nonzero_test_counts = imputed[nzind]
            nonzero_indices = vcat(nonzero_train_indices, nonzero_test_indices)
            nonzero_counts = vcat(nonzero_train, nonzero_test_counts)
        elseif mask_diag == true
            nzind_diag = findall(x -> x > 0, imputed_diag)
            nonzero_diag_indices = diag_indices[nzind_diag]
            nonzero_diag_counts = imputed_diag[nzind_diag]
            nonzero_indices = vcat(nonzero_train_indices, nonzero_diag_indices)
            nonzero_counts = vcat(nonzero_train, nonzero_diag_counts)
        else
            nonzero_indices = nonzero_train_indices
            nonzero_counts = nonzero_train
        end
        global y_M, y_Q = allocate(nonzero_indices, nonzero_counts, factor_matrices_M, indices_QM, lambdas_Q, obs_dims)  
    end
    return(y_M, y_Q)
end

function iteratively_update_factor() #update factor matrices iteratively
    for m in 1:M
        global factor_matrices_M = update_factor(y_M, factor_matrices_M, latent_dims, obs_dims, lambdas_Q, alpha, beta, indices_QM, m)   
    end
end

function impute_missing(iter, split=true, mask_diag=true) #impute heldout data
    if (split == true) &  (mod(iter, 50)==0)#imputation
        global imputed, likelihood, rates = impute(factor_matrices_M, test_indices, lambdas_Q, indices_QM, true_counts, barY) 
    end
    if (mask_diag == true) &  (mod(iter, 50)==0)#imputation
        global imputed_diag, _, _ = impute(factor_matrices_M, diag_indices, lambdas_Q, indices_QM, diag_counts, barY)
    end
end


    
function train_model(burn_in, n_iter, save_each, split=true) #train Bayesian Poisson Allocore model
    start_time = time()
    posterior_samples = []
    for iter in -burn_in:n_iter
        elapsed = @elapsed begin
            BPA(iter, split)
        end
        time_tot = time() - start_time 
        if mod(iter, save_each) == 0
        values = save_sample(sample_dir, split, iter, elapsed, time_tot)
        if (iter >= 0)
        push!(posterior_samples, values)
        end
        end
    end
    return(posterior_samples)
end



function save_sample(sample_dir, split, iter, elapsed, time_tot) #save samples
    if !isdir(sample_dir)
        mkpath(sample_dir)
    end
    if (mod(iter, save_every) == 0)
        if (split == true) & (iter > 0)
            n = iter/save_every
            global b_llks += likelihood
            global b_rates += rates
            global mae = mean(abs.(b_rates./n .- true_counts))
            global rmse = sqrt(mean((b_rates./n.-true_counts).^2))
            global nzrmse = sqrt(mean((b_rates[true_counts .> 0]./n.-true_counts[true_counts .> 0]).^2))
            global llk_value = calc_llk(b_llks, save_every, iter)
            global nz_llk_value = calc_llk(b_llks[true_counts .> 0], save_every, iter)
            values = Dict{String, Any}("factors" => factor_matrices_M, "indices" => indices_QM, "lambdas" => lambdas_Q, "time"=> elapsed, "llk" => llk_value, 
            "mae" => mae, "pis" => pis_M, "elapsed" => time_tot, "nz_llk" => nz_llk_value, "Y" => y_Q, "rmse"=>rmse, "nzrmse"=>nzrmse)
        else 
            values = Dict{String, Any}("factors" => factor_matrices_M, "indices" => indices_QM, "lambdas" => lambdas_Q, "time"=> elapsed, "pis" => pis_M, "elapsed" => time_tot, "Y"=>y_Q)
        end
        
        if mod(iter, save_every) == 0 
            save("$(sample_dir)/sample_$(iter).jld", values)
        end
        return(values)
    end
    
    if mod(iter, 100) == 0 #print progress
        println("iteration: $(iter), elapsed = $(elapsed)")
    end
    
end




function calc_llk(b_llks, save_every, iter) #calc likelihood on heldout data, given likelihood vector `b_llks`
    n = round(iter/save_every)
    avg_likelihood = b_llks ./n
    avg_likelihood[avg_likelihood.==0] .= 1e-5
    avg_likelihood[isnan.(avg_likelihood)].=1e-5
    log_avg = log.(avg_likelihood)
    value = exp(mean(log_avg))
    return(value)
end

function train_test_split(tensor, heldout, mask, obs_dims) #split data into training and test sets and diagonal core entries (for international relations data)
    test_indices = findall(x -> x == 1, heldout)
    diag_indices1 = findall(x -> x == 1, mask)
     diag_indices = setdiff(diag_indices1, test_indices)
        return(test_indices, diag_indices)
end

function gen_mask(MCAR, p, obs_dims, mask_diag) #generate mask for heldout data
    M = length(obs_dims)
    if MCAR == true
        heldout = rand(Binomial(1, p), obs_dims)
    else
        partial_heldout = rand(Binomial(1, p), obs_dims[1:M-1]) #holdout p proportion of data: 1 = heldout
        heldout = repeat(partial_heldout, outer=tuple(ones(Int, M-1)..., obs_dims[M]))
        println("observed tensor size: $(obs_dims)")
        @assert size(heldout)==obs_dims
    end

    mask = copy(heldout)
    if mask_diag == true
        for i in 1:(obs_dims[1])
            inds = ntuple(d -> d <= 2 ? i : 1:obs_dims[d], M) 
            if (M > 2)
            heldout[inds...] .= 0 #make sure that the data for heldout likelihood does not include country interactions with itself: 1: in held out evaluation set
            mask[inds...] .= 1 #1: masked 
            else
            heldout[inds...] = 0 #make sure that the data for heldout likelihood does not include country interactions with itself: 1: in held out evaluation set
            mask[inds...] = 1 #1: masked 
            end
        end
    end
    return(heldout, mask)
end


function init_model(tensor, Q, alpha=1.0,beta=10.0, epsilon=1.0, zeta=1.0, latent_dims=-1) #initialize model given parameters
    obs_dims = size(tensor)
    M = length(obs_dims)
    if latent_dims == -1 #canonical allocore
        latent_dims = fill(Q, M)
    end
    @assert length(latent_dims)==M
    options = Dict{String, Any}("Q"=> Q, "M"=> M, "latent_dims"=> latent_dims, "obs_dims"=>obs_dims)
    hyperoptions = Dict{String, Float64}("alpha" => alpha, "beta" => beta, "epsilon" => epsilon, "zeta" => zeta)
    model = Dict{String, Any}("options"=>options, "hyperoptions"=>hyperoptions)
    return(model)
end

