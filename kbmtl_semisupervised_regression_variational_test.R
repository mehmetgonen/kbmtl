# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_regression_variational_test <- function(K, state) {
  N <- dim(K)[2]
  T <- dim(state$W$mean)[2]

  H <- list(mean = crossprod(state$A$mean, K))

  Y <- list(mean = crossprod(H$mean, state$W$mean), covariance = matrix(0, N, T))
  for (t in 1:T) {
    Y$covariance[,t] <- 1 / (state$epsilon$shape[t] * state$epsilon$scale[t]) + diag(crossprod(H$mean, state$W$covariance[,,t]) %*% H$mean)
  }

  prediction <- list(H = H, Y = Y)
}