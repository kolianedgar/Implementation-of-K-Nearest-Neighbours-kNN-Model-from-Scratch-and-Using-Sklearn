# -------------------------------------------------
# Working directory (RStudio-safe)
# -------------------------------------------------
if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# -------------------------------------------------
# Load results
# -------------------------------------------------
df_own <- read.csv("knn_own_results_digits_70_30.csv", stringsAsFactors = FALSE)
df_package <- read.csv("sklearn_knn_digits_70_30.csv", stringsAsFactors = FALSE)

# -------------------------------------------------
# Extract paired Test performance metrics
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
# Monte Carlo paired permutation test
# -------------------------------------------------
paired_permutation_test_mc <- function(
    x,
    y,
    statistic = mean,
    alternative = c("greater", "less", "two.sided"),
    n_perm = 50000,
    seed = 42
) {
  alternative <- match.arg(alternative)
  set.seed(seed)
  
  diffs <- x - y
  observed <- statistic(diffs)
  n <- length(diffs)
  
  signs <- matrix(
    sample(c(-1, 1), size = n * n_perm, replace = TRUE),
    nrow = n_perm
  )
  
  perm_stats <- rowMeans(
    signs * matrix(diffs, nrow = n_perm, ncol = n, byrow = TRUE)
  )
  
  p_value <- switch(
    alternative,
    greater   = mean(perm_stats >= observed),
    less      = mean(perm_stats <= observed),
    two.sided = mean(abs(perm_stats) >= abs(observed))
  )
  
  list(
    statistic = observed,
    p_value = p_value
  )
}

# -------------------------------------------------
# Run tests for all metrics
# -------------------------------------------------
results <- lapply(names(metric_direction), function(metric) {
  
  own_vals <- own_metrics[[metric]]
  pkg_vals <- package_metrics[[metric]]
  
  # Align direction so that "greater" = sklearn better
  if (metric_direction[[metric]] == "higher") {
    x <- pkg_vals
    y <- own_vals
    alternative <- "greater"
  } else {
    x <- own_vals
    y <- pkg_vals
    alternative <- "greater"
  }
  
  perm <- paired_permutation_test_mc(
    x = x,
    y = y,
    alternative = alternative,
    n_perm = 50000
  )
  
  data.frame(
    metric = metric,
    statistic = perm$statistic,
    p_value = perm$p_value
  )
})

results_df <- do.call(rbind, results)

# -------------------------------------------------
# Export
# -------------------------------------------------
write.csv(
  results_df,
  "results/knn_mc_permutation_testing_digits_70_30.csv",
  row.names = FALSE
)

print(results_df)