# -------------------------------------------------
# Working directory (RStudio-safe)
# -------------------------------------------------
if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# -------------------------------------------------
# Load results
# -------------------------------------------------
df_own <- read.csv("knn_own_results_breast_cancer_80_20.csv", stringsAsFactors = FALSE)
df_package <- read.csv("sklearn_knn_breast_cancer_80_20.csv", stringsAsFactors = FALSE)

# -------------------------------------------------
# Extract paired metric
# -------------------------------------------------
own_ram <- df_own$fit_ram_mb
package_ram <- df_package$fit_ram_mb

stopifnot(length(own_ram) == length(package_ram))

# -------------------------------------------------
# Monte Carlo paired permutation test
# -------------------------------------------------
monte_carlo_paired_permutation <- function(
    diffs,
    statistic = mean,
    n_perm = 100000,
    alternative = "less",
    seed = 42
) {
  set.seed(seed)
  
  observed <- statistic(diffs)
  n <- length(diffs)
  
  perm_stats <- replicate(
    n_perm,
    statistic(diffs * sample(c(-1, 1), n, replace = TRUE))
  )
  
  p_value <- switch(
    alternative,
    less = mean(perm_stats <= observed),
    greater = mean(perm_stats >= observed),
    two.sided = mean(abs(perm_stats) >= abs(observed))
  )
  
  list(
    statistic = observed,
    p_value = p_value
  )
}

# -------------------------------------------------
# Run test
# -------------------------------------------------
# H1: package uses LESS RAM than own
diffs_ram <- package_ram - own_ram

perm_result <- monte_carlo_paired_permutation(
  diffs = diffs_ram,
  statistic = mean,
  n_perm = 100000,
  alternative = "less"
)

# -------------------------------------------------
# Collect results
# -------------------------------------------------
results_df <- data.frame(
  metric = "fit_ram_mb",
  statistic = perm_result$statistic,
  p_value = perm_result$p_value
)

print(results_df)

# -------------------------------------------------
# Export
# -------------------------------------------------
write.csv(
  results_df,
  file = "results/knn_mc_permuation_memory_breast_cancer_80_20.csv",
  row.names = FALSE
)

print(results_df)