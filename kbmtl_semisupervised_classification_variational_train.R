# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_classification_variational_train <- function(K, Y, parameters) {
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

  W <- list(mean = matrix(rnorm(R * T), R, T), covariance = array(diag(1, R, R), c(R, R, T)))

  F <- list(mean = (abs(matrix(rnorm(N * T), N, T)) + parameters$margin) * sign(Y), covariance = matrix(1, N, T))

  KKT <- tcrossprod(K, K)

  lower <- matrix(-1e40, N, T)
  lower[which(Y > 0)] <- +parameters$margin
  upper <- matrix(+1e40, N, T)
  upper[which(Y < 0)] <- -parameters$margin

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
      indices <- which(is.nan(Y[i,]) == FALSE)
      H$covariance[,,i] <- chol2inv(chol(diag(1 / sigmah^2, R, R) + tcrossprod(W$mean[,indices], W$mean[,indices]) + apply(W$covariance[,,indices], 1:2, sum)))
      H$mean[,i] <- H$covariance[,,i] %*% (crossprod(A$mean, K[,i]) / sigmah^2 + tcrossprod(W$mean[,indices], F$mean[i, indices, drop = FALSE]))
    }

    # update W
    for (t in 1:T) {
      indices <- which(is.nan(Y[,t]) == FALSE)
      W$covariance[,,t] <- chol2inv(chol(diag(1 / sigmaw^2, R, R) + tcrossprod(H$mean[,indices], H$mean[,indices]) + apply(H$covariance[,,indices], 1:2, sum)))
      W$mean[,t] <- W$covariance[,,t] %*% (H$mean[,indices] %*% F$mean[indices, t])
    }

    # update F
    output <- crossprod(H$mean, W$mean)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mean <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$covariance <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }

  state <- list(Lambda = Lambda, A = A, W = W, parameters = parameters)
}