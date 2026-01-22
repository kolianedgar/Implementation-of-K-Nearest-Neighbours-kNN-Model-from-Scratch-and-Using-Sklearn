# -------------------------------------------------
# Working directory (RStudio-safe)
# -------------------------------------------------
if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# -------------------------------------------------
# Load results
# -------------------------------------------------
df_own <- read.csv("knn_own_results_iris_80_20.csv", stringsAsFactors = FALSE)
df_package <- read.csv("sklearn_knn_iris_80_20.csv", stringsAsFactors = FALSE)

# -------------------------------------------------
# Extract paired CV performance metrics
# -------------------------------------------------
own_metrics <- list(
  macro_f1          = df_own$test_macro_f1,
  macro_recall      = df_own$test_macro_recall,
  macro_sensitivity = df_own$test_macro_sensitivity,
  macro_roc_auc     = df_own$test_macro_roc_auc,
  cross_entropy     = df_own$test_cross_entropy,
  brier             = df_own$test_brier,
  ece               = df_own$test_ece
)

package_metrics <- list(
  macro_f1          = df_package$test_macro_f1,
  macro_recall      = df_package$test_macro_recall,
  macro_sensitivity = df_package$test_macro_sensitivity,
  macro_roc_auc     = df_package$test_macro_roc_auc,
  cross_entropy     = df_package$test_cross_entropy,
  brier             = df_package$test_brier,
  ece               = df_package$test_ece
)

# -------------------------------------------------
# Metric direction (what is better?)
# -------------------------------------------------
metric_direction <- c(
  macro_f1          = "higher",
  macro_recall      = "higher",
  macro_sensitivity = "higher",
  macro_roc_auc     = "higher",
  cross_entropy     = "lower",
  brier             = "lower",
  ece               = "lower"
)

# -------------------------------------------------
# Wilcoxon signed-rank tests
# -------------------------------------------------
results <- lapply(names(metric_direction), function(metric) {
  
  own_vals <- own_metrics[[metric]]
  pkg_vals <- package_metrics[[metric]]
  
  # Ensure equal length and remove NA pairs
  valid <- complete.cases(own_vals, pkg_vals)
  own_vals <- own_vals[valid]
  pkg_vals <- pkg_vals[valid]
  
  alternative <- if (metric_direction[[metric]] == "higher") {
    "greater"   # package > own
  } else {
    "less"      # package < own
  }
  
  wt <- wilcox.test(
    x = pkg_vals,
    y = own_vals,
    paired = TRUE,
    alternative = alternative,
    exact = FALSE
  )
  
  data.frame(
    metric    = metric,
    statistic = unname(wt$statistic),
    p_value   = formatC(wt$p.value, format = "e", digits = 3)
  )
})

results_df <- do.call(rbind, results)

# -------------------------------------------------
# Export
# -------------------------------------------------
write.csv(
  results_df,
  "results/knn_wilcoxon_testing_iris_80_20.csv",
  row.names = FALSE
)

print(results_df)