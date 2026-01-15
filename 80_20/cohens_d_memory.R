# -------------------------------------------------
# Working directory (RStudio-safe)
# -------------------------------------------------
if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# -------------------------------------------------
# Load results
# -------------------------------------------------
df_own <- read.csv("own_knn_exhaustive_results.csv", stringsAsFactors = FALSE)
df_package <- read.csv("sklearn_knn_exhaustive_results.csv", stringsAsFactors = FALSE)

# -------------------------------------------------
# Helper function: Paired Cohen's d
# -------------------------------------------------
cohens_d_paired <- function(x_package, x_own) {
  diff <- x_package - x_own
  mean(diff) / sd(diff)
}

# -------------------------------------------------
# Extract paired metric
# -------------------------------------------------
own_ram <- df_own$fit_ram_mb
package_ram <- df_package$fit_ram_mb

# -------------------------------------------------
# Compute Cohen's d for all metrics
# -------------------------------------------------
cohens_d_results <- data.frame(
  Metric = c(
    "RAM Used for Training"
  ),
  Cohens_d = c(
    cohens_d_paired(package_ram, own_ram)
  )
)

# -------------------------------------------------
# Export
# -------------------------------------------------
write.csv(
  cohens_d_results,
  "cohens_d_memory.csv",
  row.names = FALSE
)