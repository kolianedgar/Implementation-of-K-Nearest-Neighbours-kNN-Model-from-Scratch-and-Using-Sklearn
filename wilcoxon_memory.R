memory_training_own <- c(0, 0.01171875, 0.05859375, 0.00390625, 0.3203125, 0)

memory_training_package <- c(0.0078125, 0, 0.00390625, 0, 0.01953125, 0)

wilcox_res <- wilcox.test(
  x = memory_training_own,
  y = memory_training_package,
  paired = TRUE,
  alternative = "greater",  # own > package → package uses less memory
  exact = TRUE
)

wilcox_df <- data.frame(
  metric = "memory",
  statistic_V = unname(wilcox_res$statistic),
  p_value = wilcox_res$p.value
)

if (requireNamespace("rstudioapi", quietly = TRUE)) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

write.csv(
  wilcox_df,
  file = file.path(getwd(), "knn_wilcoxon_results_memory.csv"),
  row.names = FALSE
)