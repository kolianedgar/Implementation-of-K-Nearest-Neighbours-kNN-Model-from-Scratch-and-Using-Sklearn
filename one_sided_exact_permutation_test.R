macro_avg_f1_own <- c(0.951055, 0.986078, 0.977876, 0.966976, 1, 0.924127)
macro_avg_recall_own <- c(0.965714, 0.989744, 0.978408, 0.963983, 1, 0.93)
macro_avg_sensitivity_own <- c(0.965714, 0.989744, 0.978408, 0.963983, 1, 0.93)
macro_avg_roc_auc_own <- c(0.98244, 1, 0.997467, 0.984547, 1)
categorical_cross_entropy_own <- c(0.763148, 0.089943, 0.183176, 0.309817, 0.000437, 0.130335)
multiclass_brier_own <- c(0.094444, 0.049207, 0.045938, 0.056784, 0.000239, 0.078937)
ece_own <- c(0.058333, 0.058439, 0.025059, 0.036107, 0.000359, 0.071412)

macro_avg_f1_package <- c(0.966536, 0.978877, 0.977642, 0.969288, 1, 0.849569)
macro_avg_recall_package <- c(0.966667, 0.982323, 0.977729, 0.966512, 1, 0.871429)
macro_avg_sensitivity_package <- c(0.966667, 0.982323, 0.977729, 0.966512, 1, 0.871429)
macro_avg_roc_auc_package <- c(0.966667, 0.99927, 0.997199, 0.97967, 1)
categorical_cross_entropy_package <- c(0.740477, 0.089676, 0.195958, 0.46364, 0.000175, 0.230309)
multiclass_brier_package <- c(0.07963, 0.052347, 0.043313, 0.055678, 0.000081, 0.117114)
ece_package <- c(0.052778, 0.040007, 0.028397, 0.030769, 0.00015, 0.11521)

diff_f1 <- macro_avg_f1_own - macro_avg_f1_package
diff_recall <- macro_avg_recall_own - macro_avg_recall_package
diff_sensitivity <- macro_avg_sensitivity_own - macro_avg_sensitivity_package
diff_roc_auc <- macro_avg_roc_auc_own - macro_avg_roc_auc_package
diff_cross_entropy <- categorical_cross_entropy_package - categorical_cross_entropy_own
diff_brier <- multiclass_brier_package - multiclass_brier_own
diff_ece <- ece_package - ece_own

exact_paired_permutation_test <- function(diffs, statistic = mean) {
  n <- length(diffs)
  
  signs <- expand.grid(rep(list(c(-1, 1)), n))
  perm_stats <- apply(signs, 1, function(s) statistic(s * diffs))
  
  observed_stat <- statistic(diffs)
  p_value <- mean(perm_stats >= observed_stat)
  
  list(
    statistic = observed_stat,
    p_value = p_value
  )
}

own <- list(
  f1 = macro_avg_f1_own,
  recall = macro_avg_recall_own,
  sensitivity = macro_avg_sensitivity_own,
  roc_auc = macro_avg_roc_auc_own,
  cross_entropy = categorical_cross_entropy_own,
  brier = multiclass_brier_own,
  ece = ece_own
)

package <- list(
  f1 = macro_avg_f1_package,
  recall = macro_avg_recall_package,
  sensitivity = macro_avg_sensitivity_package,
  roc_auc = macro_avg_roc_auc_package,
  cross_entropy = categorical_cross_entropy_package,
  brier = multiclass_brier_package,
  ece = ece_package
)

metric_direction <- c(
  f1 = "higher",
  recall = "higher",
  sensitivity = "higher",
  roc_auc = "higher",
  cross_entropy = "lower",
  brier = "lower",
  ece = "lower"
)

results <- lapply(names(own), function(metric) {
  
  if (metric_direction[metric] == "higher") {
    diffs <- own[[metric]] - package[[metric]]
  } else {
    diffs <- package[[metric]] - own[[metric]]
  }
  
  perm <- exact_paired_permutation_test(diffs)
  
  data.frame(
    metric = metric,
    statistic = perm$statistic,
    p_value = perm$p_value
  )
})

results_df <- do.call(rbind, results)

if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

write.csv(
  results_df,
  file = file.path(getwd(), "knn_exact_permutation_results_cv.csv"),
  row.names = FALSE
)