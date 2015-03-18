# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_classification_variational_test <- function(K, state) {
  N <- dim(K)[2]
  T <- dim(state$W$mu)[2]

  H <- list(mu = crossprod(state$A$mu, K))

  F <- list(mu = crossprod(H$mu, state$W$mu), sigma = matrix(0, N, T))
  for (t in 1:T) {
    F$sigma[,t] <- 1 + diag(crossprod(H$mu, state$W$sigma[,,t]) %*% H$mu)
  }

  pos <- 1 - pnorm((+state$parameters$margin - F$mu) / F$sigma)
  neg <- pnorm((-state$parameters$margin - F$mu) / F$sigma)
  P <- pos / (pos + neg)

  prediction <- list(H = H, F = F, P = P)
}