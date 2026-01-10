memory_training_own <- c(0, 0.01171875, 0.05859375, 0.00390625, 0.3203125, 0)

memory_training_package <- c(0.0078125, 0, 0.00390625, 0, 0.01953125, 0)

own <- list(
  memory_own = memory_training_own
)

package <- list(
  memory_package = memory_training_package
)

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

metric_direction <- c(
  memory = "lower"
)

results <- lapply(names(metric_direction), function(metric) {
  
  if (metric_direction[[metric]] == "lower") {
    diffs <- memory_training_package - memory_training_own
  } else {
    diffs <- memory_training_own - memory_training_package
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
  file = file.path(getwd(), "knn_exact_permutation_results_memory.csv"),
  row.names = FALSE
)