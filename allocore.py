import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from julia import Main
Main.include('utils.jl')
Main.include('allocore.jl')

 
class Allocore:
        """Allocore tensor decomposition with Gibbs sampling
        """

        def __init__(self, Q, latent, sharing_param, epsilon, zeta, alpha, beta):

            self.Q = Q
            self.latent = latent
            self.sharing_param = sharing_param
            self.epsilon = epsilon #shape parameter on core prior
            self.zeta = zeta #scale parameter on core prior
            self.alpha = alpha #shape parameter on factor matrix prior
            self.beta = beta #scale parameter on factor matrix prior: beta > alpha to encourage sparse factor matrices


        def fit(self, data, mask_bool=False, mask_diagonal=False, p=0.01,burn_in=100, n_iter=1000, n_thin=20, init=True):
            posterior_samples = Main.main(self.Q, self.latent, data, mask_bool, mask_diagonal, p, burn_in, n_iter, n_thin, self.sharing_param, self.epsilon, self.zeta, self.alpha, self.beta, init)
            return(posterior_samples)           
                      #verbose=True  
          
if __name__ == '__main__':
    #Example usage
    rn.seed(617)

    epsilon = 1.0 #shape parameter on core prior
    zeta = 1.0 #rate parameter on core prior
    alpha = 1.0 #shape parameter on factor prior
    beta = 10.0 #rate parameter on factor prior: beta > alpha to encourage sparse factor matrices
    sharing_param = 0.1 #dirichlet concentration parameters

    #burn_in = 100 #burn-in samples
    #n_iter = 1000 #number of iterations

    #save_each = 20 #save a sample every `save_each` sample
    

    #mask_bool=False #to holdout a random p% of M-fibers in the observed tensor
    #p = 0 #percentage of observed tensor to hold out
    #mask_diagonal=False #mask super-diagonal of observed tensor. Can be useful in some applications, e.g. network data
    

    # Generate some synthetic data with obvious block structure

    U_IK = np.concatenate([np.repeat([[5., 0.1, 0.2]], 10, axis=0),
                           np.repeat([[0.1, 5., 0.2]], 10, axis=0),
                           np.repeat([[0.1, 0.2, 3.]], 10, axis=0)])
    U_IK = rn.gamma(0.15 * U_IK, 1/0.1)

    V_KJ = np.concatenate([np.repeat([[4., 0.1, 0.2]], 30, axis=0),
                           np.repeat([[0.1, 2., 0.2]], 20, axis=0),
                           np.repeat([[0.1, 0.2, 4.]], 10, axis=0)]).T
    V_KJ = rn.gamma(0.15 * V_KJ, 1/0.1)
    
    Y_IJ = rn.poisson(U_IK @ V_KJ)
    data = Y_IJ
    #data = Main.load("uber.jld")['tensor']
    Q = 20
    latent = (10, 10)
    #latent = (5, 5, 5, 10, 10) #size of core tensor; should be same dimension as observed tensor
    #latent = (Q, Q, Q, Q) #e.g., canonical allocore with M = 4
    
    # initialize model
    model = Allocore(Q, latent, sharing_param, epsilon, zeta, alpha, beta)

    # training
    posterior_samples = model.fit(data)
    posterior_samples = list(posterior_samples)
    

    # grab last sample to inspect
    posterior_sample = posterior_samples[-1]
    Phi1_VQ, Phi2_JQ = posterior_sample['factors']
    lambdas_Q = posterior_sample['lambdas']
    indices_QM = posterior_sample['indices'] - 1 #julia indices to python indices
    Y_Q = posterior_sample['Y']

    # plot factor matrix
    unique_indices = np.bincount(indices_QM[:, 0])
    filtered = [x for x in unique_indices if x > 0]
    sns.heatmap(Phi1_VQ[:,np.unique(indices_QM[:,0])]*filtered, cmap='Blues', xticklabels = np.unique(indices_QM[:,0]))
    plt.xlabel("Factor")
    plt.ylabel("Observed Index")
    plt.title("Factor Matrix")
    plt.show()

    #plot factor matrix
    unique_indices = np.bincount(indices_QM[:, 1])
    filtered = [x for x in unique_indices if x > 0]
    sns.heatmap(Phi2_JQ[:,np.unique(indices_QM[:,1])]*filtered, cmap='Blues', xticklabels = np.unique(indices_QM[:,1]))
    plt.xlabel("Factor")
    plt.ylabel("Observed Index")
    plt.title("Factor Matrix")
    plt.show()

    # plot reconstruction
    reconstructed_IJ = (Phi1_VQ[:,indices_QM[:,0]]*lambdas_Q) @ Phi2_JQ[:,indices_QM[:,1]].T
    sns.heatmap((Phi1_VQ[:,indices_QM[:,0]]*lambdas_Q) @ Phi2_JQ[:,indices_QM[:,1]].T, cmap='Blues')
    plt.xlabel("Observed Dimension 1")
    plt.ylabel("Observed Dimension 2")
    plt.title("Reconstructed Data")
    plt.show()

    #plot data
    sns.heatmap(Y_IJ, cmap="Blues")
    plt.xlabel("Observed Dimension 1")
    plt.ylabel("Observed Dimension 2")
    plt.title("True Data")
    plt.show()

    #plot residuals
    sns.histplot((Y_IJ- reconstructed_IJ).reshape((np.prod(Y_IJ.shape),)))
    plt.xlabel("Residual")
    plt.title("Residuals")
    plt.show()



    
    df = pd.DataFrame()
    df['q'] = np.arange(indices_QM.shape[0])
    df['c'] = indices_QM[:, 0]
    df['d'] = indices_QM[:, 1]
    #df['k'] = indices_QM[:, 2]
    #df['r'] = indices_QM[:, 3]
    #df['t'] = indices_QM[:,4]
    df['lambda'] = lambdas_Q
    df['Y'] = Y_Q

    grouped_df = df.groupby(['c', 'd']).agg({'lambda': 'sum', 'Y': 'sum', 'q': list}).reset_index()
    #grouped_df = df.groupby(['c', 'd', 'k', 'r']).agg({'lambda': 'sum', 'q': list}).reset_index()
    grouped_df = grouped_df.sort_values('Y', ascending=False).reset_index(drop=True)
    
    for index, row in grouped_df.iterrows():
        lambda_q = row['lambda']
        c, d = row['c'], row['d']#, row['k'], row['r'], row['t']
        #,k, r, t
        Y = row['Y']
        Phi_V_M = [Phi1_VQ[:, c], Phi2_JQ[:,d]]
        num_plots = len(Phi_V_M)
        inds = [c,d]
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))  # Adjust figsize as needed
        colors = [sns.color_palette("Reds")[-1], sns.color_palette("Greens")[-1]]
        for i, (arr, color, ind) in enumerate(zip(Phi_V_M, colors, inds)):
            markerline, stemline, baseline = axes[i].stem(range(len(arr)), arr)
            plt.setp(baseline, 'color', color)
            plt.setp(stemline, 'color', color)
            plt.setp(markerline, 'color', color)
            plt.setp(plt.gca().get_yticklabels(), fontsize=10) #change ytick size
            axes[i].set_title(f"Factor {i+1} index: {ind}", fontsize=12)
            plt.suptitle(f'{round(Y)} allocated counts, rate: {round(lambda_q, 2)}')
            #plt.xticks(, labels, rotation=90, fontsize=10)
            #plt.savefig(title, format='pdf', bbox_inches='tight')
        plt.show()

        

        
