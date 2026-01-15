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
# Extract Test performance metrics
# -------------------------------------------------
own_data_load = df_own$dataset_load_time_s
own_cv = df_own$cv_time_s
own_fit = df_own$fit_time_s
own_test = df_own$test_time_s

package_data_load = df_package$dataset_load_time_sec
package_cv = df_package$cv_time_sec
package_fit = df_package$fit_time_sec
package_test = df_package$test_time_sec

# -------------------------------------------------
# Compute Cohen's d for all metrics
# -------------------------------------------------
cohens_d_results <- data.frame(
  Metric = c(
    "Data Load Time",
    "Cross-Validation Time",
    "Training Time",
    "Testing Time"
  ),
  Cohens_d = c(
    cohens_d_paired(package_data_load, own_data_load),
    cohens_d_paired(package_cv, own_cv),
    cohens_d_paired(package_fit, own_fit),
    cohens_d_paired(package_test, own_test)
  )
)

# -------------------------------------------------
# Export
# -------------------------------------------------
write.csv(
  cohens_d_results,
  "cohens_d_time.csv",
  row.names = FALSE
)