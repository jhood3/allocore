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
                      
          

        

        
