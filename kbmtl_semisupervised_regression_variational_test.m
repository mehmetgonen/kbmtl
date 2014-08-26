% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmtl_semisupervised_regression_variational_test(K, state)
    N = size(K, 2);
    T = size(state.W.mean, 2);

    prediction.H.mean = state.A.mean' * K;

    prediction.Y.mean = prediction.H.mean' * state.W.mean;
    prediction.Y.covariance = zeros(N, T);
    for t = 1:T
        prediction.Y.covariance(:, t) = 1 / (state.epsilon.shape(t) * state.epsilon.scale(t)) + diag(prediction.H.mean' * state.W.covariance(:, :, t) * prediction.H.mean);
    end
end