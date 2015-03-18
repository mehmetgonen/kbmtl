% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmtl_semisupervised_classification_variational_test(K, state)
    N = size(K, 2);
    T = size(state.W.mu, 2);

    prediction.H.mu = state.A.mu' * K;

    prediction.F.mu = prediction.H.mu' * state.W.mu;
    prediction.F.sigma = zeros(N, T);
    for t = 1:T
        prediction.F.sigma(:, t) = 1 + diag(prediction.H.mu' * state.W.sigma(:, :, t) * prediction.H.mu);
    end
    
    pos = 1 - normcdf((+state.parameters.margin - prediction.F.mu) ./ prediction.F.sigma);
    neg = normcdf((-state.parameters.margin - prediction.F.mu) ./ prediction.F.sigma);
    prediction.P = pos ./ (pos + neg);
end