if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

df_own <- read.csv("knn_own_results_digits_80_20.csv", stringsAsFactors = FALSE)
df_package <- read.csv("sklearn_knn_digits_80_20.csv", stringsAsFactors = FALSE)

own_data_load = df_own$dataset_load_time_s
own_cv = df_own$cv_time_s
own_fit = df_own$fit_time_s
own_test = df_own$test_time_s

package_data_load = df_package$dataset_load_time_sec
package_cv = df_package$cv_time_sec
package_fit = df_package$fit_time_sec
package_test = df_package$test_time_sec

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
  
  perm_stats <- rowMeans(signs * rep(diffs, each = n_perm))
  
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

metrics <- list(
  dataset_load_time = list(
    own = own_data_load,
    pkg = package_data_load
  ),
  cv_time = list(
    own = own_cv,
    pkg = package_cv
  ),
  fit_time = list(
    own = own_fit,
    pkg = package_fit
  ),
  test_time = list(
    own = own_test,
    pkg = package_test
  )
)


results <- lapply(names(metrics), function(metric) {
  
  own_vals <- metrics[[metric]]$own
  pkg_vals <- metrics[[metric]]$pkg
  
  perm <- paired_permutation_test_mc(
    x = pkg_vals,
    y = own_vals,
    alternative = "less",
    n_perm = 50000
  )
  
  data.frame(
    metric = metric,
    statistic = perm$statistic,
    p_value = perm$p_value
  )
})

results_df <- do.call(rbind, results)

write.csv(
  results_df,
  "results/knn_mc_permutation_time_digits_80_20.csv",
  row.names = FALSE
)

print(results_df)