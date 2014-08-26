# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmtl_semisupervised_classification_variational_test <- function(K, state) {
  N <- dim(K)[2]
  T <- dim(state$W$mean)[2]

  H <- list(mean = crossprod(state$A$mean, K))

  F <- list(mean = crossprod(H$mean, state$W$mean), covariance = matrix(0, N, T))
  for (t in 1:T) {
    F$covariance[,t] <- 1 + diag(crossprod(H$mean, state$W$covariance[,,t]) %*% H$mean)
  }

  pos <- 1 - pnorm((+state$parameters$margin - F$mean) / F$covariance)
  neg <- pnorm((-state$parameters$margin - F$mean) / F$covariance)
  P <- pos / (pos + neg)

  prediction <- list(H = H, F = F, P = P)
}