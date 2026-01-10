data_load_own <- c(
  0.07351469993591309, 0.002000570297241211, 0.014008283615112305,
  0.008049488067626953, 0.04561209678649902, 0.004076480865478516
)

hyperparameter_tuning_own <- c(
  0.18152785301208496, 0.22022175788879395, 8.409167051315308,
  0.9614145755767822, 80.15968561172485, 0.14223623275756836
)

cv_own <- c(
  0.019002676010131836, 0.02199864387512207, 0.47304797172546387,
  0.05360531806945801, 3.568678140640259, 0.02599620819091797
)

test_own <- c(
  0.004990577697753906, 0.0055348873138427734, 0.14191412925720215,
  0.015050649642944336, 1.124389410018921, 0.0147857666015625
)

data_load_package <- c(
  0.07877087593078613, 0.03752255439758301, 0.07483077049255371,
  0.0390009880065918, 0.16062331199645996, 0.025734901428222656
)

hyperparameter_tuning_package <- c(
  3.19036865234375, 0.23592376708984375, 0.9617536067962646,
  0.2297077178955078, 4.07694411277771, 0.2157883644104004
)

cv_package <- c(
  0.0404200553894043, 0.0310056209564209, 0.35585618019104004,
  0.03753042221069336, 0.6242282390594482, 0.0957183837890625
)

test_package <- c(
  0.007891178131103516, 0.011006832122802734, 0.01896190643310547,
  0.009017467498779297, 0.15751314163208008, 0.015999317169189453
)

own <- list(
  data_load = data_load_own,
  hyperparameter_tuning = hyperparameter_tuning_own,
  cv = cv_own,
  test = test_own
)

package <- list(
  data_load = data_load_package,
  hyperparameter_tuning = hyperparameter_tuning_package,
  cv = cv_package,
  test = test_package
)

wilcox_results <- lapply(names(own), function(metric) {
  
  res <- wilcox.test(
    x = own[[metric]],
    y = package[[metric]],
    paired = TRUE,
    alternative = "greater",
    exact = TRUE
  )
  
  data.frame(
    metric = metric,
    statistic_V = unname(res$statistic),
    p_value = res$p.value
  )
})

wilcox_df <- do.call(rbind, wilcox_results)

if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

write.csv(
  wilcox_df,
  file = file.path(getwd(), "knn_wilcoxon_results_time.csv"),
  row.names = FALSE
)