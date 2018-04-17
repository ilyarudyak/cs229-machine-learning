taus = [.01, .05, .1, .5, 1.0, 5.0];
[X_train,y_train] = load_data;
res = 200;
for i = 1:6
    subplot(2, 3, i);
    plot_lwlr_diff_tau(X_train, y_train, taus(i), res);
    title(['tau = ' num2str(taus(i))]);
end