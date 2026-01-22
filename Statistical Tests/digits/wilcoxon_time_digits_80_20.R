# -------------------------------------------------
# Working directory (RStudio-safe)
# -------------------------------------------------
if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# -------------------------------------------------
# Load results
# -------------------------------------------------
df_own <- read.csv("knn_own_results_digits_80_20.csv", stringsAsFactors = FALSE)
df_package <- read.csv("sklearn_knn_digits_80_20.csv", stringsAsFactors = FALSE)

# -------------------------------------------------
# Extract paired metrics
# -------------------------------------------------
own_metrics <- list(
  dataset_load_time = df_own$dataset_load_time_s,
  cv_time           = df_own$cv_time_s,
  fit_time          = df_own$fit_time_s,
  test_time         = df_own$test_time_s
)

package_metrics <- list(
  dataset_load_time = df_package$dataset_load_time_sec,
  cv_time           = df_package$cv_time_sec,
  fit_time          = df_package$fit_time_sec,
  test_time         = df_package$test_time_sec
)

# -------------------------------------------------
# Metric direction
# -------------------------------------------------
metric_direction <- c(
  dataset_load_time = "lower",
  cv_time           = "lower",
  fit_time          = "lower",
  test_time         = "lower"
)

# -------------------------------------------------
# Wilcoxon signed-rank tests (PAIRED, CORRECT)
# -------------------------------------------------
results <- lapply(names(metric_direction), function(metric) {
  
  own_vals     <- own_metrics[[metric]]
  package_vals <- package_metrics[[metric]]
  
  if (metric_direction[[metric]] == "lower") {
    # H1: package < own
    alternative <- "less"
    x <- package_vals
    y <- own_vals
  } else {
    alternative <- "greater"
    x <- own_vals
    y <- package_vals
  }
  
  test <- wilcox.test(
    x,
    y,
    paired = TRUE,
    alternative = alternative,
    exact = FALSE,
    correct = TRUE
  )
  
  data.frame(
    metric    = metric,
    statistic = unname(test$statistic),
    p_value   = test$p.value
  )
})

results_df <- do.call(rbind, results)

# -------------------------------------------------
# Export results
# -------------------------------------------------
write.csv(
  results_df,
  file = "results/knn_wilcoxon_time_digits_80_20.csv",
  row.names = FALSE
)

print(results_df)