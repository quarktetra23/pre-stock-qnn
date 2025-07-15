import numpy as np

def benchmark_comparison(avg_f1_scores):
    benchmark_scores = {1: 88.7, 2: 80.6, 3: 80.1, 5: 88.2, 10: 91.6}
    model_scores_pct = {k: avg_f1_scores[i] * 100 for i, k in enumerate([1, 2, 3, 5, 10])}
    percentage_diffs = {}

    for k in benchmark_scores:
        model_score = model_scores_pct[k]
        benchmark = benchmark_scores[k]
        diff = ((model_score - benchmark) / benchmark) * 100
        percentage_diffs[k] = diff

    avg_percentage_diff = np.mean(list(percentage_diffs.values()))

    print("\nPercentage difference from benchmark scores (positive = better, negative = worse):")
    for k in [1, 2, 3, 5, 10]:
        print(f"Horizon {k}: Our Model = {model_scores_pct[k]:.2f}% | Benchmark = {benchmark_scores[k]}% | Difference = {percentage_diffs[k]:+.2f}%")

    print(f"\nAverage % difference from benchmark across all horizons: {avg_percentage_diff:+.2f}%")
    return avg_percentage_diff
