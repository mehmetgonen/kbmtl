# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_regression_variational_train <- function(K, Y, parameters) {
  set.seed(parameters$seed)

  D <- dim(K)[1]
  N <- dim(K)[2]
  T <- dim(Y)[2]
  R <- parameters$R
  sigma_h <- parameters$sigma_h
  sigma_w <- parameters$sigma_w

  Lambda <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D, R), beta = matrix(parameters$beta_lambda, D, R))
  A <- list(mu = matrix(rnorm(D * R), D, R), sigma = array(diag(1, D, D), c(D, D, R)))
  H <- list(mu = matrix(rnorm(R * N), R, N), sigma = array(diag(1, R, R), c(R, R, N)))

  epsilon <- list(alpha = matrix(parameters$alpha_epsilon + 0.5 * colSums(!is.na(Y)), T, 1), beta = matrix(parameters$beta_epsilon, T, 1))
  W <- list(mu = matrix(rnorm(R * T), R, T), sigma = array(diag(1, R, R), c(R, R, T)))

  KKT <- tcrossprod(K, K)

  for (iter in 1:parameters$iteration) {
    # update Lambda
    for (s in 1:R) {
      Lambda$beta[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (A$mu[,s]^2 + diag(A$sigma[,,s])))
    }
    # update A
    for (s in 1:R) {
      A$sigma[,,s] <- chol2inv(chol(diag(as.vector(Lambda$alpha[,s] * Lambda$beta[,s]), D, D) + KKT / sigma_h^2))
      A$mu[,s] <- A$sigma[,,s] %*% (tcrossprod(K, H$mu[s,,drop = FALSE]) / sigma_h^2)
    }
    # update H
    for (i in 1:N) {
      indices <- which(is.na(Y[i,]) == FALSE)
      H$sigma[,,i] <- chol2inv(chol(diag(1 / sigma_h^2, R, R) + tcrossprod(W$mu[,indices, drop = FALSE], W$mu[,indices, drop = FALSE] * matrix(epsilon$alpha[indices] * epsilon$beta[indices], R, length(indices), byrow = TRUE)) + apply(W$sigma[,,indices, drop = FALSE] * array(matrix(epsilon$alpha[indices] * epsilon$beta[indices], R * R, length(indices), byrow = TRUE), c(R, R, length(indices))), 1:2, sum)))
      H$mu[,i] <- H$sigma[,,i] %*% (crossprod(A$mu, K[,i]) / sigma_h^2 + tcrossprod(W$mu[,indices, drop = FALSE], Y[i, indices, drop = FALSE] * epsilon$alpha[indices] * epsilon$beta[indices]))
    }

    # update epsilon
    for (t in 1:T) {
      indices <- which(is.na(Y[,t]) == FALSE)
      epsilon$beta[t] <- 1 / (1 / parameters$beta_epsilon + 0.5 * (crossprod(Y[indices, t, drop = FALSE], Y[indices, t, drop = FALSE]) - 2 * crossprod(Y[indices, t, drop = FALSE], crossprod(H$mu[,indices, drop = FALSE], W$mu[,t])) + sum((tcrossprod(H$mu[,indices, drop = FALSE], H$mu[,indices, drop = FALSE]) + apply(H$sigma[,,indices, drop = FALSE], 1:2, sum)) * (tcrossprod(W$mu[,t], W$mu[,t]) + W$sigma[,,t]))));
    }
    # update W
    for (t in 1:T) {
      indices <- which(is.na(Y[,t]) == FALSE)
      W$sigma[,,t] <- chol2inv(chol(diag(1 / sigma_w^2, R, R) + epsilon$alpha[t] * epsilon$beta[t] * (tcrossprod(H$mu[,indices, drop = FALSE], H$mu[,indices, drop = FALSE]) + apply(H$sigma[,,indices, drop = FALSE], 1:2, sum))))
      W$mu[,t] <- W$sigma[,,t] %*% (epsilon$alpha[t] * epsilon$beta[t] * H$mu[,indices, drop = FALSE] %*% Y[indices, t, drop = FALSE])
    }
  }

  state <- list(Lambda = Lambda, A = A, epsilon = epsilon, W = W, parameters = parameters)
}