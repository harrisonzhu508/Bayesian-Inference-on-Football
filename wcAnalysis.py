#The aim of this project is to predict the scores head-to-head matches of world cup 2018 in Russia.
#In particular, we work under the Bayesian framework, using the Poisson regression likelihood with a normalised Gaussian prior. Our 3
#approximations will be the Laplace approximation, Metropolis-Hastings and Gaussian Variational Approximation (GVA). Our prediction is
#made using sampling. In particular, using the large law of numbers and the ergodic central limit theorem.
#Our dataset contains head-to-head match scores between the 32 teams from 1995 to 2018. We distinguish between home and away teams. 

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
train_input = pd.read_csv("/Users/harrisonzhu/Desktop/World Cup/wc.csv")
#label the 32 teams from 0 to 31
Teams_target = sorted(list(set(train_input.localTeam)))
Teams_main = pd.DataFrame(data = {"Teams" : Teams_target})
Teams = {}
for i in range(len(Teams_main)):
    Teams[Teams_main.Teams[i]] = i

#----------------------------------------------------------------------------------------------------------------------------------
#Here we develop the functions we will use for our analysis
#Define the Posterior and its derivatives
#likelihood function
def likelihood(theta, train_input):
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

def posterior(theta, train_input):
    prior = e ** (-0.5 * theta.norm() ** 2)

    return likelihood(theta, train_input) * prior



#Simple log unnormalised-posterior we call log_MAP
#train_input is our dataframe
def log_MAP(theta, train_input):
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
        L = -( e ** (theta[0] + theta[1+indexH] - theta[32+indexA]) + e ** (theta[0] + theta[1+indexA] - theta[32+indexH]) ) + float(train_input.localGoals[i]) * (theta[1+indexH] - theta[32+indexA]) + float(train_input.visitorGoals[i]) * (theta[1+indexA] - theta[32+indexH])
        somme2 = somme2 + L

    return(somme2)

def gradient_log_MAP(theta, train_input):
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
        somme1 = somme1 - e ** (Delta + alphas[indexH] - betas[indexA]) - e ** (alphas[indexA] - betas[indexH]) + float(train_input.visitorGoals[i])
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
#we use the minibatch stochastic gradient descent

def SGD_batch(train, log_MAP, gradient_log_MAP, theta, lr = 1e-1, nb_epochs = 250):
    #initialisation
    train_input = train
    #load up the criterion and gradient function
    criterion = log_MAP
    gradient = gradient_log_MAP

    #number of epochs for training
    #nb_epochs = 250
    n = len(train) #number of samples
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

#Markov chain sampling approach

#define the acceptance ratio
def posterior_ratio(theta_proposal, theta_start, train_input):

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
def posterior_ratio_log(theta_proposal, theta_start, train_input):
    
    val = log_MAP(theta_proposal, train_input) - log_MAP(theta_start, train_input)
    
    return e ** val
    
#the metroplis-hastings algorithm function
def MH_RW_tuned(n, theta0, train_input, lamda, batch_size):
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
            ratio = posterior_ratio_log(theta_proposal, theta_start, train_input)
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
        ratio = posterior_ratio_log(theta_proposal, theta_start, train_input)
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

#Laplace_sample samples from what we obtain from the laplace approximation method
def Laplace_sample(mu, covariance, nb_sample1, nb_sample2):

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

#similar to the laplace sampler, we skip the step where we sample thetas from the posterior
def chain_predictor(theta_samples, nb_sample1, nb_sample2):

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
                    team2 = sum(np.
                                random.poisson(mu_ij_away, nb_sample2))

                    Samples_storage[i][j] = Samples_storage[i][j] + Tensor([float(team1), float(team2)])

    #take the average of all the sample scores 
    Samples_storage = Samples_storage / (nb_sample1 * nb_sample2)
    return Samples_storage

#Gaussian Variational Approximation (GVA)

#define the Evidence Lower Bound (ELBO)
def ELBO(data, mu_n, L_n):
    #data is a tensor storing the thetas
    #mu_n, L_n initialised at some points
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
    
    return(somme)

def gradmu_ELBO(data, mu_n, L_n):
    l = 100
    d = len(mu_n)
    e_L = Tensor(scipy.linalg.expm(L_n.numpy()))
    somme = 0
    
    for i in range(l):
        eta = Tensor(d,1).normal_(0,1)
        Eeta = e_L.mm(eta)
        somme = somme + gradient_log_MAP(Eeta + mu_n, data)
    somme = -somme/l
    
    
    return(somme)

#the gradient of the ELBO using the Hessian approach
def gradL_ELBO1(data, mu_n, L_n):
    
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
    
    return(somme)

#the gradient of the ELBO using the gradient approach
def gradL_ELBO2(data, mu_n, L_n):
    
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
    
    return(somme)

#The GVA via the gradient ELBO gradient
def GVA_grad(initial_vals, data, lr, plot_yes = True):
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

#predict head-to-head score function
def results(hometeam, awayteam, scores_mat):
    #hometeam, awayteam as strings
    indexh = 2*Teams[hometeam]
    indexa = Teams[awayteam]
    
    homegoal = scores_mat[indexa, indexh]
    awaygoal = scores_mat[indexa, indexh + 1]
    
    return hometeam + ":" + '{:0.2f}'.format(homegoal) + " " + awayteam + ":" + '{:0.2f}'.format(awaygoal)    
#----------------------------------------------------------------------------------------------------------------------------------

