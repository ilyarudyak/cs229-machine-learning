function [clusters, centroids] = k_means(X, k)

%%% YOUR CODE HERE

% choose k random different Xi as initial centroids
rng(42);
centroids = X(randperm(size(X, 1), k), :);
m = size(X, 1);

s = 1;
while s <= 10
    % for each point find closest centroid
    XR = repmat(X, [1 1 k]);
    mur = repmat(reshape(centroids', 1, 2, k), m, 1);
    [M, clusters] = min(vecnorm(XR - mur, 2, 2), [], k);

    draw_clusters(X, clusters, centroids);
    pause(1);
    
    % recompute centroids
    for i = 1:k
        centroids(i, :) = mean(X(clusters == i, :));
    end
    s = s + 1;
end