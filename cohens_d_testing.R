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
macro_f1_own          <- df_own$test_macro_f1
macro_recall_own      <- df_own$test_macro_recall
macro_sensitivity_own <- df_own$test_macro_sensitivity
macro_roc_auc_own     <- df_own$test_macro_roc_auc
cross_entropy_own     <- df_own$test_cross_entropy
brier_own             <- df_own$test_brier
ece_own               <- df_own$test_ece

macro_f1_package          <- df_package$test_macro_f1
macro_recall_package      <- df_package$test_macro_recall
macro_sensitivity_package <- df_package$test_macro_sensitivity
macro_roc_auc_package     <- df_package$test_macro_roc_auc
cross_entropy_package     <- df_package$test_cross_entropy
brier_package             <- df_package$test_brier
ece_package               <- df_package$test_ece

# -------------------------------------------------
# Compute Cohen's d for all metrics
# -------------------------------------------------
cohens_d_results <- data.frame(
  Metric = c(
    "Macro F1",
    "Macro Recall",
    "Macro Sensitivity",
    "Macro ROC-AUC",
    "Cross-Entropy",
    "Brier Score",
    "Expected Calibration Error"
  ),
  Cohens_d = c(
    cohens_d_paired(macro_f1_package, macro_f1_own),
    cohens_d_paired(macro_recall_package, macro_recall_own),
    cohens_d_paired(macro_sensitivity_package, macro_sensitivity_own),
    cohens_d_paired(macro_roc_auc_package, macro_roc_auc_own),
    cohens_d_paired(cross_entropy_package, cross_entropy_own),
    cohens_d_paired(brier_package, brier_own),
    cohens_d_paired(ece_package, ece_own)
  )
)

# -------------------------------------------------
# Export
# -------------------------------------------------
write.csv(
  cohens_d_results,
  "cohens_d_testing.csv",
  row.names = FALSE
)
