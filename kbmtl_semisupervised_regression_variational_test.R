# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_regression_variational_test <- function(K, state) {
  N <- dim(K)[2]
  T <- dim(state$W$mu)[2]

  H <- list(mu = crossprod(state$A$mu, K))

  Y <- list(mu = crossprod(H$mu, state$W$mu), sigma = matrix(0, N, T))
  for (t in 1:T) {
    Y$sigma[,t] <- 1 / (state$epsilon$alpha[t] * state$epsilon$beta[t]) + diag(crossprod(H$mu, state$W$sigma[,,t]) %*% H$mu)
  }

  prediction <- list(H = H, Y = Y)
}