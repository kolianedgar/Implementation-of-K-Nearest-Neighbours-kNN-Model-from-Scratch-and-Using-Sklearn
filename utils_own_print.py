def print_cv_results(results, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for metric, (mean, std) in results.items():
        print(f"{metric:<30}: {mean:.6f} ± {std:.6f}")
        
def print_test_results(results):
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric:<30}: {value:.6f}")
