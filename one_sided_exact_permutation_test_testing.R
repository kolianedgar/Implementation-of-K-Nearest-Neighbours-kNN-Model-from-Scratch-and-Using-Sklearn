macro_avg_f1_own <- c(0.93266, 1, 0.966143, 0.971277, 1, 1)
macro_avg_recall_own <- c(0.933333, 1, 0.966424, 0.964286, 1, 1)
macro_avg_sensitivity_own <- c(0.933333, 1, 0.966424, 0.964286, 1, 1)
macro_avg_roc_auc_own <- c(0.993333, 1, 0.995024, 0.982143, 1, 1)
categorical_cross_entropy_own <- c(0.113787, 0.078654, 0.283232, 0.31153, 0, 0.038969)
multiclass_brier_own <- c(0.081481, 0.036711, 0.048, 0.05657, 0, 0.017331)
ece_own <- c(0.011111, 0.066801, 0.025556, 0.042607, 0, 0.031381)

macro_avg_f1_package <- c(0.93266, 1, 0.966143, 0.961911, 1, 1)
macro_avg_recall_package <- c(0.933333, 1, 0.966424, 0.957341, 1, 1)
macro_avg_sensitivity_package <- c(0.933333, 1, 0.966424, 0.957341, 1, 1)
macro_avg_roc_auc_package <- c(0.993333, 1, 0.995006, 0.971396, 1, 1)
categorical_cross_entropy_package <- c(0.113787, 0.069276, 0.282578, 0.536039, 0, 0.148487)
multiclass_brier_package <- c(0.081481, 0.030612, 0.047334, 0.068226, 0, 0.064417)
ece_package <- c(0.011111, 0.059524, 0.026812, 0.032164, 0, 0.110333)

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
  file = file.path(getwd(), "knn_exact_permutation_results_testing.csv"),
  row.names = FALSE
)