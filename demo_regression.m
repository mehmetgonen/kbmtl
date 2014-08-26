%initalize the parameters of the algorithm
parameters = struct();

%set the hyperparameters of gamma prior used for projection matrix
parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;

%set the hyperparameters of gamma prior used for output noise
parameters.alpha_epsilon = 1;
parameters.beta_epsilon = 1;

%%% IMPORTANT %%%
%For gamma priors, you can experiment with three different (alpha, beta) values
%(1, 1) => default priors
%(1e-10, 1e+10) => good for obtaining sparsity
%(1e-10, 1e-10) => good for small sample size problems

%set the number of iterations
parameters.iteration = 200;

%set the subspace dimensionality
parameters.R = 20;

%set the seed for random number generator used to initalize random variables
parameters.seed = 1606;

%set the standard deviation of hidden representations
parameters.sigmah = 0.1;

%set the standard deviation of weight parameters
parameters.sigmaw = 1.0;

%initialize the kernel and target outputs for training
Ktrain = ??; %should be an Ntra x Ntra matrix containing similarity values between training samples
Ytrain = ??; %should be an Ntra x T matrix containing target outputs of tasks (contains only real values and NaNs)

%perform training
state = kbmtl_semisupervised_regression_variational_train(Ktrain, Ytrain, parameters);

%initialize the kernel for testing
Ktest = ??; %should be an Ntra x Ntest matrix containing similarity values between training and test samples

%perform prediction
prediction = kbmtl_semisupervised_regression_variational_test(Ktest, state);

%display the predictions
display(prediction.Y.mean);
