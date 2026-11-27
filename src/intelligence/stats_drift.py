# ==============================================================================
# WHAT: Statistical Anomaly & Distribution Drift Engine [Topic 02: Math & Stats]
# WHY:  Production GCP telemetry contains heavy-tailed spikes and unequal variances.
#       Relying on simple thresholds causes alert fatigue. This module mathematically
#       validates whether an observed metric shift or data-plane anomaly is a
#       statistically significant incident before invoking ML/DL/LLM layers.
# HOW:  Implements three complementary statistical tests:
#       1. Modified Z-Score (using Median + MAD) -> outlier-resistant spike check.
#       2. Welch's Two-Sample t-test -> hypothesis test for unequal variances.
#       3. KL-Divergence -> measures probability distribution drift vs. baseline.
# WHERE: `src/intelligence/stats_drift.py` (called by the MCP diagnostics tool
#        and Streamlit UI diagnostic cards).
# EXAMPLE:
#   from src.intelligence.stats_drift import StatisticalDriftAnalyzer
#   analyzer = StatisticalDriftAnalyzer()
#   report = analyzer.evaluate_signal(observed_value=98.7, baseline_value=64.0)
# ==============================================================================

from typing import Dict, List                                               # Imports type hints for structured dictionary and list returns
import numpy as np                                                          # Imports NumPy for fast vectorized array math and percentiles
from scipy import stats                                                     # Imports SciPy statistical tests (Welch's t-test and entropy/KL)


class StatisticalDriftAnalyzer:                                             # Encapsulates all mathematical anomaly and drift checks for OmniRCA-GCP
    def __init__(                                                           # Initializes statistical decision thresholds
        self,
        z_threshold: float = 3.5,                                           # Standard Iglewicz-Hoaglin threshold for Modified Z-Score outliers
        p_value_alpha: float = 0.05,                                        # 95% confidence level threshold for Welch's t-test significance
        kl_threshold: float = 0.15,                                         # Divergence cutoff indicating meaningful distribution shift
    ):
        self.z_threshold = z_threshold                                      # Stores Modified Z-Score cutoff
        self.p_value_alpha = p_value_alpha                                  # Stores hypothesis test p-value cutoff
        self.kl_threshold = kl_threshold                                    # Stores KL-Divergence drift cutoff

    def compute_modified_z_score(                                           # Computes outlier-resistant Z-score using Median and MAD
        self, observed_value: float, baseline_series: List[float]
    ) -> float:
        arr = np.asarray(baseline_series, dtype=float)                      # Converts baseline list into a NumPy float array
        median = float(np.median(arr))                                      # Computes the 50th percentile (immune to extreme spikes)
        mad = float(np.median(np.abs(arr - median)))                        # Computes Median Absolute Deviation (median distance from median)
        if mad == 0.0:                                                      # Guards against division by zero if baseline values are constant
            mad = max(abs(median) * 0.05, 1e-6)                             # Uses 5% of median (or tiny epsilon) as fallback scale
        mod_z = 0.6745 * (observed_value - median) / mad                    # Applies 0.6745 scaling constant so MAD matches normal std dev
        return round(float(mod_z), 4)                                       # Returns rounded Modified Z-Score

    def compute_welch_ttest(                                                # Tests whether incident window mean differs from baseline mean
        self, baseline_series: List[float], incident_series: List[float]
    ) -> Dict[str, float | bool]:
        t_stat, p_val = stats.ttest_ind(                                    # Runs two-sample t-test
            incident_series,                                                # Active incident sample values
            baseline_series,                                                # Historical healthy baseline sample values
            equal_var=False,                                                # Sets equal_var=False to run Welch's t-test (handles bursty variance)
        )
        t_clean = 0.0 if np.isnan(t_stat) else float(t_stat)                # Replaces NaN with 0.0 if both series are identical constants
        p_clean = 1.0 if np.isnan(p_val) else float(p_val)                  # Replaces NaN p-value with 1.0 (no evidence of difference)
        return {                                                            # Returns structured hypothesis test results
            "t_statistic": round(t_clean, 4),                               # Magnitude and direction of shift in standard errors
            "p_value": round(p_clean, 6),                                   # Probability that this shift happened purely by random chance
            "is_significant": bool(p_clean < self.p_value_alpha),           # True if p < 0.05 (statistically confirmed anomaly)
        }

    def compute_kl_divergence(                                              # Measures how much incident distribution diverged from baseline
        self, baseline_series: List[float], incident_series: List[float], bins: int = 10
    ) -> float:
        combined = np.concatenate([baseline_series, incident_series])       # Combines both arrays so histogram bin edges align identically
        bin_edges = np.histogram_bin_edges(combined, bins=bins)             # Computes shared bin boundaries across the combined range
        p_counts, _ = np.histogram(baseline_series, bins=bin_edges)         # Counts baseline observations per bin
        q_counts, _ = np.histogram(incident_series, bins=bin_edges)         # Counts incident observations per bin
        p_dist = (p_counts + 1e-6) / np.sum(p_counts + 1e-6)                # Smooths with 1e-6 (Laplace smoothing) and normalizes to probabilities
        q_dist = (q_counts + 1e-6) / np.sum(q_counts + 1e-6)                # Ensures no zero probabilities (prevents log(0) = infinity)
        kl_val = float(stats.entropy(qk=q_dist, pk=p_dist))                 # Computes KL(Q || P) = sum(Q * log(Q / P))
        return round(kl_val, 4)                                             # Returns non-negative divergence score (0.0 = identical distributions)

    def evaluate_signal(                                                    # High-level convenience method evaluating a single IncidentSignal
        self, observed_value: float, baseline_value: float, seed: int = 42
    ) -> Dict[str, float | bool]:
        rng = np.random.default_rng(seed)                                   # Creates deterministic random generator for reproducible baseline window
        base_scale = max(abs(baseline_value) * 0.08, 0.5)                   # Simulates realistic +/-8% operational jitter around healthy baseline
        inc_scale = max(abs(observed_value) * 0.05, 0.5)                    # Simulates +/-5% jitter around the observed incident state
        baseline_window = rng.normal(loc=baseline_value, scale=base_scale, size=30).tolist() # Generates 30 historical healthy baseline points
        incident_window = rng.normal(loc=observed_value, scale=inc_scale, size=10).tolist()  # Generates 10 active incident window points

        mod_z = self.compute_modified_z_score(observed_value, baseline_window) # Calculates outlier-resistant Modified Z-Score
        ttest_res = self.compute_welch_ttest(baseline_window, incident_window) # Runs Welch's two-sample t-test
        kl_div = self.compute_kl_divergence(baseline_window, incident_window)  # Calculates distribution KL-Divergence

        is_confirmed = (                                                    # Combines all 3 mathematical checks into a final anomaly verdict
            abs(mod_z) >= self.z_threshold                                  # True if spike exceeds 3.5 MAD units
            and bool(ttest_res["is_significant"])                           # AND shift is statistically significant (p < 0.05)
        )
        return {                                                            # Returns complete mathematical audit dictionary
            "observed_value": observed_value,                               # Echoes observed metric value
            "baseline_value": baseline_value,                               # Echoes healthy baseline value
            "modified_z_score": mod_z,                                      # Modified Z-Score (> 3.5 or < -3.5 = strong outlier)
            "welch_t_stat": ttest_res["t_statistic"],                       # Welch's t-statistic
            "welch_p_value": ttest_res["p_value"],                          # Welch's p-value
            "kl_divergence": kl_div,                                        # KL-Divergence drift score
            "is_confirmed_anomaly": is_confirmed,                           # Final boolean verdict filtering out false-alarm noise
        }