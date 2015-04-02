% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmtl_semisupervised_regression_variational_train(K, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(K, 1);
    N = size(K, 2);
    T = size(Y, 2);
    R = parameters.R;
    sigma_h = parameters.sigma_h;
    sigma_w = parameters.sigma_w;

    Lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(D, R);
    Lambda.beta = parameters.beta_lambda * ones(D, R);
    A.mu = randn(D, R);
    A.sigma = repmat(eye(D, D), [1, 1, R]);
    H.mu = randn(R, N);
    H.sigma = repmat(eye(R, R), [1, 1, N]);

    epsilon.alpha = (parameters.alpha_epsilon + 0.5 * sum(~isnan(Y))');
    epsilon.beta = parameters.beta_epsilon * ones(T, 1);
    W.mu = randn(R, T);
    W.sigma = repmat(eye(R, R), [1, 1, T]);

    KKT = K * K';

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update Lambda
        for s = 1:R
            Lambda.beta(:, s) = 1 ./ (1 / parameters.beta_lambda + 0.5 * (A.mu(:, s).^2 + diag(A.sigma(:, :, s))));
        end
        %%%% update A
        for s = 1:R
            A.sigma(:, :, s) = (diag(Lambda.alpha(:, s) .* Lambda.beta(:, s)) + KKT / sigma_h^2) \ eye(D, D);
            A.mu(:, s) = A.sigma(:, :, s) * (K * H.mu(s, :)' / sigma_h^2);
        end
        %%%% update H
        for i = 1:N
            indices = ~isnan(Y(i, :));
            H.sigma(:, :, i) = (eye(R, R) / sigma_h^2 + W.mu(:, indices) * (W.mu(:, indices)' .* repmat(epsilon.alpha(indices) .* epsilon.beta(indices), 1, R)) + sum(W.sigma(:, :, indices) .* reshape(repmat(epsilon.alpha(indices) .* epsilon.beta(indices), 1, R * R)', [R, R, sum(indices)]), 3)) \ eye(R, R);
            H.mu(:, i) = H.sigma(:, :, i) * (A.mu' * K(:, i) / sigma_h^2 + W.mu(:, indices) * (Y(i, indices)' .* epsilon.alpha(indices) .* epsilon.beta(indices)));
        end

        %%%% update epsilon
        for t = 1:T
            indices = ~isnan(Y(:, t));
            epsilon.beta(t) = 1 / (1 / parameters.beta_epsilon + 0.5 * (Y(indices, t)' * Y(indices, t) - 2 * Y(indices, t)' * H.mu(:, indices)' * W.mu(:, t) + sum(sum((H.mu(:, indices) * H.mu(:, indices)' + sum(H.sigma(:, :, indices), 3)) .* (W.mu(:, t) * W.mu(:, t)' + W.sigma(:, :, t))))));
        end
        %%%% update W
        for t = 1:T
            indices = ~isnan(Y(:, t));
            W.sigma(:, :, t) = (eye(R, R) / sigma_w^2 + epsilon.alpha(t) * epsilon.beta(t) * (H.mu(:, indices) * H.mu(:, indices)' + sum(H.sigma(:, :, indices), 3))) \ eye(R, R);
            W.mu(:, t) = W.sigma(:, :, t) * (epsilon.alpha(t) * epsilon.beta(t) * H.mu(:, indices) * Y(indices, t));
        end
    end

    state.Lambda = Lambda;
    state.A = A;
    state.epsilon = epsilon;
    state.W = W;
    state.parameters = parameters;
end