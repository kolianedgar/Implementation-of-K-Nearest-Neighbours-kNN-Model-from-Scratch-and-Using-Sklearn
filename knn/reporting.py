def print_cv_results(results, title):
    """
        Print cross-validation results in a formatted table.

        Displays the mean and standard deviation of each metric
        obtained from cross-validation in a readable layout.

        Parameters
        ----------
        results : dict
            Dictionary mapping metric names to ``(mean, std)``
            tuples, typically produced by a cross-validation
            routine.

        title : str
            Title displayed above the results table.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for metric, (mean, std) in results.items():
        print(f"{metric:<30}: {mean:.6f} ± {std:.6f}")
        
def print_test_results(results):
    """
        Print evaluation results on a test dataset.

        Displays final metric values computed on a held-out
        test set in a clear, formatted layout.

        Parameters
        ----------
        results : dict
            Dictionary mapping metric names to scalar values
            computed on the test dataset.
    """
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric:<30}: {value:.6f}")
