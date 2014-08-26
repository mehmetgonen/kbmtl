% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmtl_semisupervised_classification_variational_train(K, Y, parameters)
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

    W.mean = randn(R, T);
    W.covariance = repmat(eye(R, R), [1, 1, T]);

    F.mean = (abs(randn(N, T)) + parameters.margin) .* sign(Y);
    F.covariance = ones(N, T);

    KKT = K * K';

    lower = -1e40 * ones(N, T);
    lower(Y > 0) = +parameters.margin;
    upper = +1e40 * ones(N, T);
    upper(Y < 0) = -parameters.margin;

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
            H.covariance(:, :, i) = (eye(R, R) / sigmah^2 + W.mean(:, indices) * W.mean(:, indices)' + sum(W.covariance(:, :, indices), 3)) \ eye(R, R);
            H.mean(:, i) = H.covariance(:, :, i) * (A.mean' * K(:, i) / sigmah^2 + W.mean(:, indices) * F.mean(i, indices)');
        end

        %%%% update W
        for t = 1:T
            indices = ~isnan(Y(:, t));
            W.covariance(:, :, t) = (eye(R, R) / sigmaw^2 + H.mean(:, indices) * H.mean(:, indices)' + sum(H.covariance(:, :, indices), 3)) \ eye(R, R);
            W.mean(:, t) = W.covariance(:, :, t) * (H.mean(:, indices) * F.mean(indices, t));
        end

        %%%% update F
        output = H.mean' * W.mean;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        F.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        F.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
    end

    state.Lambda = Lambda;
    state.A = A;
    state.W = W;
    state.parameters = parameters;
end