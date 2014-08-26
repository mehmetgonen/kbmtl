% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmtl_semisupervised_regression_variational_train(K, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(K, 1);
    N = size(K, 2);
    T = size(Y, 2);
    R = parameters.R;
    sigmah = parameters.sigmah;
    sigmaw = parameters.sigmaw;

    Lambda.shape = (parameters.alpha_lambda + 0.5) * ones(D, R);
    Lambda.scale = parameters.beta_lambda * ones(D, R);
    A.mean = randn(D, R);
    A.covariance = repmat(eye(D, D), [1, 1, R]);
    H.mean = randn(R, N);
    H.covariance = repmat(eye(R, R), [1, 1, N]);

    epsilon.shape = (parameters.alpha_epsilon + 0.5 * sum(~isnan(Y))');
    epsilon.scale = parameters.beta_epsilon * ones(T, 1);
    W.mean = randn(R, T);
    W.covariance = repmat(eye(R, R), [1, 1, T]);

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
            Lambda.scale(:, s) = 1 ./ (1 / parameters.beta_lambda + 0.5 * (A.mean(:, s).^2 + diag(A.covariance(:, :, s))));
        end
        %%%% update A
        for s = 1:R
            A.covariance(:, :, s) = (diag(Lambda.shape(:, s) .* Lambda.scale(:, s)) + KKT / sigmah^2) \ eye(D, D);
            A.mean(:, s) = A.covariance(:, :, s) * (K * H.mean(s, :)' / sigmah^2);
        end
        %%%% update H
        for i = 1:N
            indices = ~isnan(Y(i, :));
            H.covariance(:, :, i) = (eye(R, R) / sigmah^2 + W.mean(:, indices) * (W.mean(:, indices)' .* repmat(epsilon.shape(indices) .* epsilon.scale(indices), 1, R)) + sum(W.covariance(:, :, indices) .* reshape(repmat(epsilon.shape(indices) .* epsilon.scale(indices), 1, R * R)', [R, R, sum(indices)]), 3)) \ eye(R, R);
            H.mean(:, i) = H.covariance(:, :, i) * (A.mean' * K(:, i) / sigmah^2 + W.mean(:, indices) * (Y(i, indices)' .* epsilon.shape(indices) .* epsilon.scale(indices)));
        end

        %%%% update epsilon
        for t = 1:T
            indices = ~isnan(Y(:, t));
            epsilon.scale(t) = 1 / (1 / parameters.beta_epsilon + 0.5 * (Y(indices, t)' * Y(indices, t) - 2 * Y(indices, t)' * H.mean(:, indices)' * W.mean(:, t) ...
                                                                         + sum(sum((H.mean(:, indices) * H.mean(:, indices)' + sum(H.covariance(:, :, indices), 3)) .* (W.mean(:, t) * W.mean(:, t)' + W.covariance(:, :, t))))));
        end
        %%%% update W
        for t = 1:T
            indices = ~isnan(Y(:, t));
            W.covariance(:, :, t) = (eye(R, R) / sigmaw^2 + epsilon.shape(t) * epsilon.scale(t) * (H.mean(:, indices) * H.mean(:, indices)' + sum(H.covariance(:, :, indices), 3))) \ eye(R, R);
            W.mean(:, t) = W.covariance(:, :, t) * (epsilon.shape(t) * epsilon.scale(t) * H.mean(:, indices) * Y(indices, t));
        end
    end

    state.Lambda = Lambda;
    state.A = A;
    state.epsilon = epsilon;
    state.W = W;
    state.parameters = parameters;
end