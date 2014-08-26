% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmtl_semisupervised_classification_variational_test(K, state)
    N = size(K, 2);
    T = size(state.W.mean, 2);

    prediction.H.mean = state.A.mean' * K;

    prediction.F.mean = prediction.H.mean' * state.W.mean;
    prediction.F.covariance = zeros(N, T);
    for t = 1:T
        prediction.F.covariance(:, t) = 1 + diag(prediction.H.mean' * state.W.covariance(:, :, t) * prediction.H.mean);
    end
    
    pos = 1 - normcdf((+state.parameters.margin - prediction.F.mean) ./ prediction.F.covariance);
    neg = normcdf((-state.parameters.margin - prediction.F.mean) ./ prediction.F.covariance);
    prediction.P = pos ./ (pos + neg);
end