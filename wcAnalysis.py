"""
The aim of this project is to predict the scores head-to-head matches of world cup 2018 in Russia.

In particular, we work under the Bayesian framework, using the Poisson regression likelihood with a normalised Gaussian prior. Our 3
approximations will be the:

- Laplace approximation, 
- Metropolis-Hastings and 
- Gaussian Variational Approximation (GVA). 

Our prediction is made using sampling. In particular, using Monte Carlo integration.

Our dataset contains head-to-head match scores between the 32 teams from 1995 to 2018. We distinguish between home and away teams. 
"""

#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from math import e, pi, log, factorial
import scipy
from scipy.stats import poisson

#import pytorch libraries
import torch
from torch import Tensor


#load the data
#training data
train_input = pd.read_csv("./wc.csv")
#label the 32 teams from 0 to 31
Teams_target = sorted(list(set(train_input.localTeam)))
Teams_main = pd.DataFrame(data = {"Teams" : Teams_target})
Teams = {}
for i in range(len(Teams_main)):
    Teams[Teams_main.Teams[i]] = i

#----------------------------------------------------------------------------------------------------------------------------------
#Define the un_Posterior and its derivatives
#Here we develop the functions we will use for our analysis
def likelihood(theta, train_input):
    """
    likelihood function

    Input:

        theta: model parameters
        train_input: training set

    Output:

        prod: the likelihood

    """
    prod = 1
    #collect coefficients
    Delta = theta[0]
    alphas = theta[1:32]
    betas = theta[32:63]

    for i in range(len(train_input)):
        indexH = Teams[train_input.localTeam[i]]
        indexA = Teams[train_input.visitorTeam[i]]
        y_home = float(train_input.localGoals[i])
        y_away = float(train_input.visitorGoals[i])

        mu_home = e ** (Delta + alphas[indexH] - betas[indexA])
        mu_away = e ** (alphas[indexA] - betas[indexH])

        first_prob = poisson.pmf(y_home, mu_home)
        second_prob = poisson.pmf(y_away, mu_away)

        prod = prod*first_prob*second_prob

    return prod

def un_posterior(theta, train_input):
    """
    Define the unnormalised posterior distribution

    input:

        theta: model parameters
        train_input: training set

    output:

        unPosterior: unnormalised posterior
    """
    prior = e ** (-0.5 * theta.norm() ** 2)
    unPosterior = likelihood(theta, train_input) * prior

    return unPosterior



def log_MAP(theta, train_input):
    """
    Simple log unnormalised-posterior we call log_MAP

    input:

        theta: model parameters
        train_input: training set

    output:

        somme2: log unnormalised-posterior
    """
    #collect coefficients
    #Delta = theta[0]
    #alphas = theta[1:21]
    #betas = theta[21:41]
    
    #calculate regularisor
    regularisor = -0.5 * theta.norm() ** 2
    
    #second sum
    somme2 = regularisor
    for i in range(len(train_input)):
        indexH = Teams[train_input.localTeam[i]]
        indexA = Teams[train_input.visitorTeam[i]]

        L = -( e ** (theta[0] + theta[1+indexH] - theta[32+indexA]) \
            + e ** (theta[0] + theta[1+indexA] - theta[32+indexH]) ) \
            + float(train_input.localGoals[i]) * (theta[1+indexH] - theta[32+indexA]) \
            + float(train_input.visitorGoals[i]) * (theta[1+indexA] - theta[32+indexH])

        somme2 = somme2 + L

    return somme2

def gradient_log_MAP(theta, train_input):
    """
    Gradient of the log unnormalised-posterior

    input:

        theta: model parameters
        train_input: training set

    output:

        gradient: gradient of the log-posterior
    """
    #collect coefficients
    Delta = theta[0]
    alphas = theta[1:32]
    betas = theta[32:63]
    
    #grad vector
    gradient = torch.zeros(63,1)   
    
    #first component with respect to Delta
    somme1 = -Delta
    for i in range(len(train_input)):
        indexH = Teams[train_input.localTeam[i]]
        indexA = Teams[train_input.visitorTeam[i]]
        somme1 = somme1 - e ** (Delta + alphas[indexH] - betas[indexA]) \
                - e ** (alphas[indexA] - betas[indexH]) + float(train_input.visitorGoals[i])
    gradient[0] = float(somme1)

    #components
    somme2 = 0
    somme3 = 0

    for k in range(31):
        somme2 = -alphas[k]
        somme3 = -betas[k]
        for i in range(len(train_input)):
            indexH = Teams[train_input.localTeam[i]]
            indexA = Teams[train_input.visitorTeam[i]]
            if indexH == k:
                somme2 = somme2 - e ** (Delta + alphas[indexH] - betas[indexA]) + float(train_input.localGoals[i])
                somme3 = somme3 + e ** (alphas[indexA] - betas[indexH]) - float(train_input.visitorGoals[i])
            if indexA == k:
                somme2 = somme2 - e ** (alphas[indexA] - betas[indexH]) + float(train_input.visitorGoals[i])
                somme3 = somme3 + e ** (Delta + alphas[indexH] - betas[indexA]) - float(train_input.localGoals[i])

        gradient[1+k] = float(somme2)
        gradient[32+k] = float(somme3)

    return gradient


def Hess_log_MAP(theta, train_input):
    """
    The hessian of the log posterior
    
    input:

        theta: model parameters
        train_input: training set

    output:

        Tensor(Hess): Hessian of the log posterior
    """
    #collect coefficients
    Delta = theta[0]
    alphas = theta[1:32]
    betas = theta[32:63]

    #grad vector
    Hess = np.zeros([63, 63])
    
    #first component with respect to Delta
    somme1 = -1
    
    for k in range(len(train_input)):
        indexH = Teams[train_input.localTeam[k]]
        indexA = Teams[train_input.visitorTeam[k]]
        somme1 = somme1 - e ** (Delta + alphas[indexH] - betas[indexA]) - e ** (alphas[indexA] - betas[indexH])
    
    Hess[0][0] = float(somme1)

    #2nd components with respect to delta alphai and alphai^2 and betai^2
    somme2 = 0
    somme3 = 0
    somme4 = -1
    somme5 = -1
    
    for k in range(31):
        somme2 = 0
        somme3 = 0
        somme4 = -1
        somme5 = -1
        for i in range(len(train_input)):
            indexH = Teams[train_input.localTeam[i]]
            indexA = Teams[train_input.visitorTeam[i]]

            if indexH == k:
                somme2 = somme2 - e ** (Delta + alphas[indexH] - betas[indexA])
                somme3 = somme3 - e ** (alphas[indexA] - betas[indexH])
                somme4 = somme4 - e ** (Delta + alphas[indexH] - betas[indexA])
                somme5 = somme5 - e ** (alphas[indexA] - betas[indexH])
            if indexA == k:
                somme2 = somme2 - e ** (alphas[indexA] - betas[indexH])
                somme3 = somme3 - e ** (Delta + alphas[indexH] - betas[indexA])
                somme4 = somme4 - e ** (alphas[indexA] - betas[indexH])    
                somme5 = somme5 - e ** (Delta + alphas[indexH] - betas[indexA])     

        Hess[0][1 + k] = float(somme2)
        Hess[1 + k][0] = float(somme2)

        Hess[0][32 + k] = float(somme3)
        Hess[32 + k][0] = float(somme3)

        Hess[1 + k][1 + k] = float(somme4)
        Hess[32 + k][32 + k] = float(somme5)

    #6th components with respect to alphai betai
    somme6 = 0
    
    for k in range(31):
        for l in range(31):
            somme6 = 0
            for i in range(len(train_input)):
                indexH = Teams[train_input.localTeam[i]]
                indexA = Teams[train_input.visitorTeam[i]]

                if indexH == k & indexA == l:
                    somme6 = somme6 + e ** (Delta + alphas[indexH] - betas[indexA])

                if indexH == l & indexA == k:
                    somme6 = somme6 + e ** (alphas[indexA] - betas[indexH])     

            Hess[1 + k][32 + l] = float(somme6)
            Hess[32 + l][1 + k] = float(somme6)

    return Tensor(Hess)


#optimisation tools

def SGD_batch(train_input, log_MAP, gradient_log_MAP, theta, lr = 1e-1, nb_epochs = 250):
    """
    We use the minibatch stochastic gradient descent

    input:

        train_input:  training set
        log_MAP:  the log-posterior function
        gradient_log_MAP: the gradient of the log-posterior function
        theta:  parameters of our model
        lr = 1e-1:  learning rate (descent rate)
        nb_epochs = 250:    number of training epochs 

    output:

        theta:  maximum a posteriori (MAP) estimator
        record_loss: the training loss
    """
    #initialisation
    train_input
    #load up the criterion and gradient function
    criterion = log_MAP
    gradient = gradient_log_MAP

    #number of epochs for training
    #nb_epochs = 250
    n = len(train_input) #number of samples
    mini_batch_size = n // 20
    sum_loss = 0
    count_of_decrease = 0
    record_loss = np.zeros([nb_epochs])

    #gradient descent
    for e in range(0, nb_epochs):

        sum_loss = 0
        for b in range(0, n, mini_batch_size):
            theta = theta + lr * gradient(theta, train_input[b:b + mini_batch_size].reset_index())
            loss = -criterion(theta, train_input[b:b + mini_batch_size].reset_index())
            sum_loss = sum_loss + loss

        record_loss[e] = loss
        if e > 0 and record_loss[e - 1] < record_loss[e]:
            count_of_decrease += 1
        if count_of_decrease == 5:
            lr = lr * 0.5
            count_of_decrease = 0


    plt.plot(record_loss)
    plt.show()

    return theta, record_loss

#Markov chain Monte Carlo sampling approach

def un_posterior_ratio(theta_proposal, theta_start, train_input):
    """
    Define the acceptance ratio

    input:
        
        theta_proposal: value obtained by proposal distribution
        theta_start: the current value of the Markov chain
        train_input: training set

    output:

        acceptance ratio


    """

    #collect coefficients
    Deltaprop = theta_proposal[0]
    alphasprop = theta_proposal[1:21]
    betasprop = theta_proposal[21:41]

    Deltastart = theta_start[0]
    alphasstart = theta_start[1:21]
    betasstart = theta_start[21:41]

    #prior
    prod1 = e ** ( -0.5 * (theta_proposal.norm() ** 2 - theta_start.norm() ** 2))

    #likelihood
    prod = 1
    for i in range(len(train_input)):

        indexH = Teams[train_input.localTeam[i]]
        indexA = Teams[train_input.visitorTeam[i]]
        y_home = float(train_input.localGoals[i])
        y_away = float(train_input.visitorGoals[i])

        mu_homeprop = e ** (Deltaprop + alphasprop[indexH] - betasprop[indexA])
        mu_homestart = e ** (Deltastart + alphasstart[indexH] - betasstart[indexA])
        mu_awayprop = e ** (alphasprop[indexA] - betasprop[indexH])
        mu_awaystart = e ** (alphasstart[indexA] - betasstart[indexH])


        ratio1 = (mu_homeprop/mu_homestart) ** y_home
        ratio2 = (mu_awayprop/mu_awaystart) ** y_home

        prod = prod * ratio1 * ratio2

    return prod1 * prod

#define the log of the acceptance ratio
def un_posterior_ratio_log(theta_proposal, theta_start, train_input):
    """
    Define the log of the acceptance ratio and then calculate the acceptance ratio

    input:
        
        theta_proposal: value obtained by proposal distribution
        theta_start: the current value of the Markov chain
        train_input: training set

    output:

        acceptance ratio
    """


    val = log_MAP(theta_proposal, train_input) - log_MAP(theta_start, train_input)
    
    return e ** val
    
def MH_RW_tuned(n, theta0, train_input, lamda, batch_size):
    """
    The random walk Metroplis-Hastings algorithm with uniform proposal

    input:
    
    n: number of samples
    theta0: initial value in the Markov chain
    train_input: training set
    lamda: factor of the size of the walk
    batch_size: batch size to select a lamda such that it gives a good acceptance probability

    output:

        Samples_storage: samples of a Markov chain
    """
    
    #for this algorithm, we should start from a relatively small, negative value of lambda
    theta0 = theta0.view(-1)
    #d = dimension of theta
    d = len(theta0)
    Chain = torch.zeros([batch_size + 1, d])
    Chain[0] = theta0
    Ratio = np.zeros([batch_size + 1])
    Ratio[0] = 0
    C = 0
    lamda_final = lamda
    theta_start = theta0

    #batch loop to find a good lamda
    while C/batch_size < 0.1 or C/batch_size > 0.5:
        C = 0
        for i in range(batch_size):
            theta_proposal = theta_start + Tensor(np.random.uniform(-1,1,d)*lamda_final)
            #rejection/acceptance
            ratio = un_posterior_ratio_log(theta_proposal, theta_start, train_input)
            p = min(1, ratio)
            #rejection step
            bern = np.random.binomial(1,p,1)
            if bern == 1:
                theta_start = theta_proposal
                C = C + 1

            Chain[i+1] = theta_start
            Ratio[i+1] = ratio    
            lamda_final = lamda_final + 0.00001

    print(C/batch_size)

    #official loop
    #reinitialise everything
    Chain = np.zeros([n+1,d])
    Chain[0] = theta0
    Ratio = np.zeros([n+1])
    Ratio[0] = 0
    C = 0
    lamda_final = lamda
    theta_start = theta0

    for i in range(n):
        theta_proposal = theta_start + Tensor(np.random.uniform(-1,1,d)*lamda_final)
        #rejection/acceptance
        ratio = un_posterior_ratio_log(theta_proposal, theta_start, train_input)
        p = min(1,ratio)

        #rejection step
        bern = np.random.binomial(1,p,1)
        if bern == 1:
            theta_start = theta_proposal
            C = C + 1
        Chain[i+1] = theta_start
        Ratio[i+1] = ratio

    return Chain, np.mean(Ratio), C/n

#Prediction tools
def Laplace_sample(mu, covariance, nb_sample1, nb_sample2):
    """
    Laplace_sample samples from what we obtain from the laplace approximation method and the likelihood.
    Then we use these to perform prediction via Monte Carlo integration

    input:
        
        mu: estimated mean of the Gaussian from the Laplace approximation
        covariance: estimated covariance matrix of the Gaussian from the Laplace approximation
        nb_sample1: number of samples from the posterior
        nb_sample2: number of samples from the likelihood

    output:

        Samples_storage: stores the sample from the posterior and likelihood (dependent on the sample from posterior)
    """

    #step 1: generate nb_sample samples of theta
    theta_samples = Tensor(np.random.multivariate_normal(mu.view(-1), covariance, nb_sample1))

    #step 2: for each of the 380 needed predictions, generate 100 samples from each of the 1000
    #samples of theta
    #initialisation of storage matrix and prediction variables
    Samples_storage = torch.zeros(31, 31, 1, 2)

    #calculate this matrix full of samples
    for i in range(31):
        for j in range(31):
            if j != i:
                for k in range(nb_sample1):
                    #parameters
                    thetak = theta_samples[k]
                    delta = thetak[0]
                    alphai = thetak[1 + i]
                    alphaj = thetak[1 + j]
                    betai = thetak[32 + i]
                    betaj = thetak[32 + j]
                    mu_ij_home = e ** (delta + alphai - betaj)
                    mu_ij_away = e ** (alphaj - betai)

                    team1 = sum(np.random.poisson(mu_ij_home, nb_sample2))
                    team2 = sum(np.random.poisson(mu_ij_away, nb_sample2))


                    Samples_storage[i][j] = Samples_storage[i][j] + Tensor([float(team1), float(team2)])

    #take the average of all the sample scores
    Samples_storage = Samples_storage / (nb_sample1 * nb_sample2)
    return Samples_storage

def chain_predictor(theta_samples, nb_sample1, nb_sample2):
    """
    Similar to the laplace sampler, we skip the step where we sample thetas from the unnormalised posterior
    Then we use these to perform prediction via Monte Carlo integration

    input:
        
        theta_samples: samples obtained from RW Metropolis-Hastings
        nb_sample1: number of samples from the posterior
        nb_sample2: number of samples from the likelihood

    output:

        Samples_storage: stores the sample from the posterior and likelihood (dependent on the sample from posterior)
    """

    #step 1: for each of the 380 needed predictions, generate 100 samples from each of the 1000
    #samples of theta
    #initialisation of storage matrix and prediction variables
    Samples_storage = torch.zeros(20, 20, 1, 2)

    #calculate this matrix full of samples
    for i in range(20):
        for j in range(20):
            if j != i:
                for k in range(nb_sample1):
                    
                    #parameters
                    thetak = theta_samples[k]
                    delta = thetak[0]
                    alphai = thetak[1 + i]
                    alphaj = thetak[1 + j]
                    betai = thetak[21 + i]
                    betaj = thetak[21 + j]
                    mu_ij_home = e ** (delta + alphai - betaj)
                    mu_ij_away = e ** (alphaj - betai)

                    team1 = sum(np.random.poisson(mu_ij_home, nb_sample2))
                    team2 = sum(np.random.poisson(mu_ij_away, nb_sample2))

                    Samples_storage[i][j] = Samples_storage[i][j] + Tensor([float(team1), float(team2)])

    #take the average of all the sample scores 
    Samples_storage = Samples_storage / (nb_sample1 * nb_sample2)
    return Samples_storage

#Gaussian Variational Approximation (GVA)
def ELBO(data, mu_n, L_n):
    """
    Define the Evidence Lower Bound (ELBO).

    input:
       
        data: is a tensor storing the thetas
        mu_n: initialised at some points
        L_n: initialised at some points

    output:

        somme: the  ELBO


    """
    
    #let's say the variance covariance matrix is the identity
    l = 100
    d = len(mu_n)
    e_L = Tensor(scipy.linalg.expm(L_n.numpy()))
    somme = + d/2 * log(2 * pi * e) + np.trace(L_n)
    
    for i in range(l):
        eta = Tensor(d,1).normal_(0,1)
        Eeta = e_L.mm(eta)
        somme = somme + log_MAP(Eeta + mu_n, data)
    somme = -somme/l
    
    return somme

def gradmu_ELBO(data, mu_n, L_n):
    """
    Compute the gradient of the ELBO with respect to mu

    input:

        data: is a tensor storing the thetas
        mu_n: initialised at some points
        L_n: initialised at some points

    output:

        somme: gradient of the ELBO

    """
    l = 100
    d = len(mu_n)
    e_L = Tensor(scipy.linalg.expm(L_n.numpy()))
    somme = 0
    
    for i in range(l):
        eta = Tensor(d,1).normal_(0,1)
        Eeta = e_L.mm(eta)
        somme = somme + gradient_log_MAP(Eeta + mu_n, data)
    somme = -somme/l
    
    return somme

def gradL_ELBO1(data, mu_n, L_n):
    """
    The gradient of the ELBO with respect to the matrix L using the Hessian approach

    input:
        
        data: is a tensor storing the thetas
        mu_n: initialised at some points
        L_n: initialised at some points

    output:

        somme: gradient of the ELBO with respect to the matrix L
    """
    
    l = 100
    d = len(mu_n)
    somme = torch.eye(d)
    e_2L = Tensor(scipy.linalg.expm(2 * L_n.numpy()))
    e_L = Tensor(scipy.linalg.expm(L_n.numpy()))
    
    for i in range(l):
        eta = Tensor(d,1).normal_(0,1)
        Eeta = e_L.mm(eta)
        somme = somme + Tensor(Hess_log_MAP(mu_n + Eeta, train_input))        
    somme = e_2L.mm(somme / l)
    somme = -0.5 * (somme + somme.t())
    somme = somme + torch.eye(d)
    
    return somme

def gradL_ELBO2(data, mu_n, L_n):
    """
    The gradient of the ELBO using the gradient approach 
    
    input:
        
        data: is a tensor storing the thetas
        mu_n: initialised at some points
        L_n: initialised at some points

    output:

        somme: gradient of the ELBO with respect to the matrix L

    """
    
    l = 100
    d = len(mu_n)
    somme = torch.eye(d)
    e_L = Tensor(scipy.linalg.expm(L_n.numpy()))
    
    for i in range(l):
        eta = Tensor(d,1).normal_(0,1)
        Eeta = e_L.mm(eta)
        somme = somme + Tensor(Eeta.t().mm(gradient_log_MAP(mu_n + Eeta, train_input)))        
    somme = somme / l
    somme = -0.5 * (somme + somme.t())
    somme = somme + torch.eye(d)
    
    return somme

def GVA_grad(initial_vals, data, lr, plot_yes = True):
    """
    The GVA via the gradient ELBO gradient

    initial_vals: initial values of mu_n and L_n
    data: is a tensor storing the thetas
    lr: the learning rate or descent rate
    plot_yes: Default = True. Plots the ELBO

    output:

        The computed mu_n and L_n

    """
    
    #initialisation
    #d = dimension of mu
    #l = length of data
    l = 100
    mu_n = initial_vals[0]
    L_n = initial_vals[1]
    d = len(mu_n)
    nb_epochs = 100
    energy = np.zeros([100])
    #print("Target", mu_0, varcov_matrix)
    count_of_decrease = 0
    
    for k in range(nb_epochs):
        
        gradL = gradL_ELBO2(data, mu_n, L_n)
        gradmu = gradmu_ELBO(data, mu_n, L_n)
        energy[k] = ELBO(data, mu_n, L_n)
        
        if energy[k] < energy[k - 1] and k > 0:
            count_of_decrease += 1
            
        if count_of_decrease > 4:
            count_of_decrease = 0
            lr = lr * 0.5

        #update
        mu_n = mu_n + lr * gradmu
        L_n = L_n + lr * gradL

    if plot_yes:
        plt.plot(energy)
        plt.show()

    return mu_n, L_n

def results(hometeam, awayteam, scores_mat):
    """
    Predict head-to-head score function

    input:

        hometeam: name of the home team as string
        awayteam: name of the away team as string
        scores_mat: stores the scores

    output:

        The result of the game
    """

    #hometeam, awayteam as strings
    indexh = 2*Teams[hometeam]
    indexa = Teams[awayteam]
    
    homegoal = scores_mat[indexa, indexh]
    awaygoal = scores_mat[indexa, indexh + 1]
    
    return hometeam + ":" + '{:0.2f}'.format(homegoal) + " " + awayteam + ":" + '{:0.2f}'.format(awaygoal)    
#----------------------------------------------------------------------------------------------------------------------------------

