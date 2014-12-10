# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_regression_variational_train <- function(K, Y, parameters) {
  set.seed(parameters$seed)

  D <- dim(K)[1]
  N <- dim(K)[2]
  T <- dim(Y)[2]
  R <- parameters$R
  sigmah <- parameters$sigmah
  sigmaw <- parameters$sigmaw

  Lambda <- list(shape = matrix(parameters$alpha_lambda + 0.5, D, R), scale = matrix(parameters$beta_lambda, D, R))
  A <- list(mean = matrix(rnorm(D * R), D, R), covariance = array(diag(1, D, D), c(D, D, R)))
  H <- list(mean = matrix(rnorm(R * N), R, N), covariance = array(diag(1, R, R), c(R, R, N)))

  epsilon <- list(shape = matrix(parameters$alpha_epsilon + 0.5 * colSums(!is.na(Y)), T, 1), scale = matrix(parameters$beta_epsilon, T, 1))
  W <- list(mean = matrix(rnorm(R * T), R, T), covariance = array(diag(1, R, R), c(R, R, T)))

  KKT <- tcrossprod(K, K)

  for (iter in 1:parameters$iteration) {
    # update Lambda
    for (s in 1:R) {
      Lambda$scale[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (A$mean[,s]^2 + diag(A$covariance[,,s])))
    }
    # update A
    for (s in 1:R) {
      A$covariance[,,s] <- chol2inv(chol(diag(as.vector(Lambda$shape[,s] * Lambda$scale[,s]), D, D) + KKT / sigmah^2))
      A$mean[,s] <- A$covariance[,,s] %*% (tcrossprod(K, H$mean[s,,drop = FALSE]) / sigmah^2)
    }
    # update H
    for (i in 1:N) {
      indices <- which(is.na(Y[i,]) == FALSE)
      H$covariance[,,i] <- chol2inv(chol(diag(1 / sigmah^2, R, R) + tcrossprod(W$mean[,indices], W$mean[,indices] * matrix(epsilon$shape[indices] * epsilon$scale[indices], R, length(indices), byrow = TRUE)) + apply(W$covariance[,,indices] * array(matrix(epsilon$shape[indices] * epsilon$scale[indices], R * R, length(indices), byrow = TRUE), c(R, R, length(indices))), 1:2, sum)))
      H$mean[,i] <- H$covariance[,,i] %*% (crossprod(A$mean, K[,i]) / sigmah^2 + tcrossprod(W$mean[,indices], Y[i, indices, drop = FALSE] * epsilon$shape[indices] * epsilon$scale[indices]))
    }

    # update epsilon
    for (t in 1:T) {
      indices <- which(is.na(Y[,t]) == FALSE)
      epsilon$scale[t] <- 1 / (1 / parameters$beta_epsilon + 0.5 * (crossprod(Y[indices, t], Y[indices, t]) - 2 * crossprod(Y[indices, t], crossprod(H$mean[,indices], W$mean[,t])) + sum((tcrossprod(H$mean[,indices], H$mean[,indices]) + apply(H$covariance[,,indices], 1:2, sum)) * (tcrossprod(W$mean[,t], W$mean[,t]) + W$covariance[,,t]))));
    }
    # update W
    for (t in 1:T) {
      indices <- which(is.na(Y[,t]) == FALSE)
      W$covariance[,,t] <- chol2inv(chol(diag(1 / sigmaw^2, R, R) + epsilon$shape[t] * epsilon$scale[t] * (tcrossprod(H$mean[,indices], H$mean[,indices]) + apply(H$covariance[,,indices], 1:2, sum))))
      W$mean[,t] <- W$covariance[,,t] %*% (epsilon$shape[t] * epsilon$scale[t] * H$mean[,indices] %*% Y[indices, t])
    }
  }

  state <- list(Lambda = Lambda, A = A, epsilon = epsilon, W = W, parameters = parameters)
}