def print_cv_results(results, title):
    """
        Print cross-validation results in a formatted, human-readable form.

        Parameters
        ----------
        results : dict
            Dictionary mapping metric names to tuples of (mean, standard deviation).
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
        Print final test set evaluation results in a formatted, human-readable form.

        Parameters
        ----------
        results : dict
            Dictionary mapping metric names to scalar metric values.
    """

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric:<30}: {value:.6f}")