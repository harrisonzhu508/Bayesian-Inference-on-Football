# Bayesian-Football-Prediction

README to be perfected soon :) 

The data are head-to-head results of international matches between the World Cup 2018 teams. The models we use are Gaussian priors and

- Logistic Regression with logit link likelihood
- Poisson Regression with log link likelihood
- Poisson Regression with soft-ReLU link likelihood

Our approximation techniques are

- Laplace approximation
- Gaussian variational approximation
- Metropolis algorithm

To predict, we use sampling methods to calculate the integrals - crude Monte Carlo integration.

## Code

We use PyTorch for its efficient tensor operations, but for purpose of education we will not use the efficient backpropogation methods. 

``wcAnalysis.py`` contains all the functions needed for all 3 approximations and its corresponding sampling functions
``test.py`` runs the algorithm and runs the analysis
``worldcup.ipynb`` was my development platform when I tested out my analysis. All the functions have now been properly implemented in ``wcAnalysis.py``