macro_avg_f1_own <- c(0.932660, 1.000000, 0.966143, 0.971277, 1.000000, 1.000000)
macro_avg_recall_own <- c(0.933333, 1.000000, 0.966424, 0.964286, 1.000000, 1.000000)
macro_avg_sensitivity_own <- c(0.933333, 1.000000, 0.966424, 0.964286, 1.000000, 1.000000)
macro_avg_roc_auc_own <- c(0.993333, 1.000000, 0.995024, 0.982143, 1.000000, 1.000000)
categorical_cross_entropy_own <- c(0.113787, 0.078654, 0.283232, 0.311530, -0.000000, 0.038969)
multiclass_brier_own <- c(0.081481, 0.036711, 0.048000, 0.056570, 0.000000, 0.017331)
ece_own <- c(0.011111, 0.066801, 0.025556, 0.042607, 0.000000, 0.031381)

macro_avg_f1_package <- c(0.932660, 1.000000, 0.966143, 0.961911, 1.000000, 1.000000)
macro_avg_recall_package <- c(0.933333, 1.000000, 0.966424, 0.957341, 1.000000, 1.000000)
macro_avg_sensitivity_package <- c(0.933333, 1.000000, 0.966424, 0.957341, 1.000000, 1.000000)
macro_avg_roc_auc_package <- c(0.993333, 1.000000, 0.995006, 0.971396, 1.000000, 1.000000)
categorical_cross_entropy_package <- c(0.113787, 0.069276, 0.282578, 0.536039, -0.000000, 0.148487)
multiclass_brier_package <- c(0.081481, 0.030612, 0.047334, 0.068226, 0.000000, 0.064417)
ece_package <- c(0.011111, 0.059524, 0.026812, 0.032164, 0.000000, 0.110333)

diff_f1 <- macro_avg_f1_own - macro_avg_f1_package
diff_recall <- macro_avg_recall_own - macro_avg_recall_package
diff_sensitivity <- macro_avg_sensitivity_own - macro_avg_sensitivity_package
diff_roc_auc <- macro_avg_roc_auc_own - macro_avg_roc_auc_package
diff_cross_entropy <- categorical_cross_entropy_package - categorical_cross_entropy_own
diff_brier <- multiclass_brier_package - multiclass_brier_own
diff_ece <- ece_package - ece_own

exact_paired_permutation_test <- function(diffs, statistic = mean) {
  n <- length(diffs)
  
  # All possible sign combinations
  signs <- expand.grid(rep(list(c(-1, 1)), n))
  
  # Permutation distribution
  perm_stats <- apply(signs, 1, function(s) statistic(s * diffs))
  
  observed_stat <- statistic(diffs)
  
  # One-sided p-value: P(stat >= observed)
  p_value <- mean(perm_stats >= observed_stat)
  
  list(
    statistic = observed_stat,
    p_value = p_value
  )
}

result_perm_f1 <- exact_paired_permutation_test(diff_f1)
result_perm_roc_auc <- exact_paired_permutation_test(diff_roc_auc)
result_perm_recall <- exact_paired_permutation_test(diff_recall)
result_perm_sensitivity <- exact_paired_permutation_test(diff_sensitivity)
result_perm_cross_entropy <- exact_paired_permutation_test(diff_cross_entropy)
result_perm_brier <- exact_paired_permutation_test(diff_brier)
result_perm_ece <- exact_paired_permutation_test(diff_ece)

result_perm_ece$statistic
result_perm_ece$p_value
