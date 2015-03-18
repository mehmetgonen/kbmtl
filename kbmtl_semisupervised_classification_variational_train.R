# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_classification_variational_train <- function(K, Y, parameters) {
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

  W <- list(mu = matrix(rnorm(R * T), R, T), sigma = array(diag(1, R, R), c(R, R, T)))

  F <- list(mu = (abs(matrix(rnorm(N * T), N, T)) + parameters$margin) * sign(Y), sigma = matrix(1, N, T))

  KKT <- tcrossprod(K, K)

  lower <- matrix(-1e40, N, T)
  lower[which(Y > 0)] <- +parameters$margin
  upper <- matrix(+1e40, N, T)
  upper[which(Y < 0)] <- -parameters$margin

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
      H$sigma[,,i] <- chol2inv(chol(diag(1 / sigma_h^2, R, R) + tcrossprod(W$mu[,indices, drop = FALSE], W$mu[,indices, drop = FALSE]) + apply(W$sigma[,,indices, drop = FALSE], 1:2, sum)))
      H$mu[,i] <- H$sigma[,,i] %*% (crossprod(A$mu, K[,i]) / sigma_h^2 + tcrossprod(W$mu[,indices, drop = FALSE], F$mu[i, indices, drop = FALSE]))
    }

    # update W
    for (t in 1:T) {
      indices <- which(is.na(Y[,t]) == FALSE)
      W$sigma[,,t] <- chol2inv(chol(diag(1 / sigma_w^2, R, R) + tcrossprod(H$mu[,indices, drop = FALSE], H$mu[,indices, drop = FALSE]) + apply(H$sigma[,,indices, drop = FALSE], 1:2, sum)))
      W$mu[,t] <- W$sigma[,,t] %*% (H$mu[,indices] %*% F$mu[indices, t, drop = FALSE])
    }

    # update F
    output <- crossprod(H$mu, W$mu)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mu <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$sigma <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }

  state <- list(Lambda = Lambda, A = A, W = W, parameters = parameters)
}