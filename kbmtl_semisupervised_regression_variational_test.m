% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmtl_semisupervised_regression_variational_test(K, state)
    N = size(K, 2);
    T = size(state.W.mu, 2);

    prediction.H.mu = state.A.mu' * K;

    prediction.Y.mu = prediction.H.mu' * state.W.mu;
    prediction.Y.sigma = zeros(N, T);
    for t = 1:T
        prediction.Y.sigma(:, t) = 1 / (state.epsilon.alpha(t) * state.epsilon.beta(t)) + diag(prediction.H.mu' * state.W.sigma(:, :, t) * prediction.H.mu);
    end
end
