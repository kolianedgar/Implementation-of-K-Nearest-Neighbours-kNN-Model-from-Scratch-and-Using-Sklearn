# -------------------------------------------------
# Working directory (RStudio-safe)
# -------------------------------------------------
if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# -------------------------------------------------
# Load results
# -------------------------------------------------
df_own <- read.csv("knn_own_results_zoo_80_20.csv", stringsAsFactors = FALSE)
df_package <- read.csv("sklearn_knn_zoo_80_20.csv", stringsAsFactors = FALSE)

# -------------------------------------------------
# Extract paired metric
# -------------------------------------------------
own_ram <- df_own$fit_ram_mb
package_ram <- df_package$fit_ram_mb

stopifnot(length(own_ram) == length(package_ram))

# -------------------------------------------------
# Wilcoxon signed-rank test (paired)
# -------------------------------------------------
# H1: package uses LESS RAM than own
wilcox_res <- wilcox.test(
  x = package_ram,
  y = own_ram,
  paired = TRUE,
  alternative = "less",
  exact = FALSE,     # correct for n = 240
  correct = FALSE    # no continuity correction
)

# -------------------------------------------------
# Collect results
# -------------------------------------------------
results_df <- data.frame(
  metric = "fit_ram_mb",
  statistic = wilcox_res$statistic,
  p_value = wilcox_res$p.value
)

print(results_df)

# -------------------------------------------------
# Export
# -------------------------------------------------
write.csv(
  results_df,
  file = "results/knn_wilcoxon_memory_zoo_80_20.csv",
  row.names = FALSE
)

print(results_df)