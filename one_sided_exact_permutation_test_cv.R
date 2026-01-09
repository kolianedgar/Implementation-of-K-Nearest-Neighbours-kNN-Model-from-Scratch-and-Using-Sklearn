macro_avg_f1_own <- c(0.951055, 0.986078, 0.977876, 0.966976, 1.000000, 0.924127)
macro_avg_recall_own <- c(0.965714, 0.989744, 0.978408, 0.963983, 1.000000, 0.930000)
macro_avg_sensitivity_own <- c(0.965714, 0.989744, 0.978408, 0.963983, 1.000000, 0.930000)
macro_avg_roc_auc_own <- c(0.982440, 1.000000, 0.997467, 0.984547, 1.000000)
categorical_cross_entropy_own <- c(0.763148, 0.089943, 0.183176, 0.309817, 0.000437, 0.130335)
multiclass_brier_own <- c(0.094444, 0.049207, 0.045938, 0.056784, 0.000239, 0.078937)
ece_own <- c(0.058333, 0.058439, 0.025059, 0.036107, 0.000359, 0.071412)

macro_avg_f1_package <- c(0.966536, 0.978877, 0.977642, 0.969288, 1.000000, 0.849569)
macro_avg_recall_package <- c(0.965714, 0.982323, 0.977729, 0.966512, 1.000000, 0.871429)
macro_avg_sensitivity_package <- c(0.965714, 0.982323, 0.977729, 0.966512, 1.000000, 0.871429)
macro_avg_roc_auc_package <- c(0.978125, 0.999270, 0.997199, 0.979670, 1.000000, 0.995238)
categorical_cross_entropy_package <- c(0.740477, 0.089676, 0.195958, 0.463640, 0.000175, 0.230309)
multiclass_brier_package <- c(0.079630, 0.052347, 0.043313, 0.055678, 0.000081, 0.117114)
ece_package <- c(0.052778, 0.040007, 0.028397, 0.030769, 0.000150, 0.115210)

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
