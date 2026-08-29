import numpy as np
import matplotlib.pyplot as plt

# Define window centers (normalized Levenshtein distance)
window_centers = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Mandarin has too few samples in the first two windows, so start from 0.2-0.4 (center 0.3).
mandarin_window_centers = window_centers[3:]

# Dummy RMSE values per language and configuration
# Mandarin: word relatedness features should help less due to different script/language family
rmse = {
    "German": {
        "B":    np.array([0.99, 1.11, 1.16, 1.18, 1.21, 1.18, 1.12, 1.07, 1.06]),
        "BE":   np.array([0.97, 1.02, 1.07, 1.09, 1.11, 1.09, 1.03, 1.00, 0.99]),
        "BS":   np.array([0.88, 1.01, 1.07, 1.12, 1.14, 1.12, 1.05, 1.01, 1.01]),
        "BR":   np.array([0.93, 1.01, 1.05, 1.10, 1.13, 1.11, 1.04, 1.01, 1.02]),
        "BES":  np.array([0.86, 0.99, 1.02, 1.08, 1.09, 1.08, 1.03, 0.99, 0.99]),
        "BSR":  np.array([0.88, 1.00, 1.05, 1.10, 1.12, 1.10, 1.04, 1.00, 1.01]),
        "BER":  np.array([0.95, 1.00, 1.05, 1.07, 1.10, 1.07, 1.02, 0.99, 0.99]),
        "BESR": np.array([0.85, 0.96, 1.02, 1.06, 1.08, 1.07, 1.02, 0.99, 0.99]),
    },
    "Spanish": {
        "B":    np.array([1.06, 1.18, 1.22, 1.24, 1.22, 1.20, 1.17, 1.16, 1.15]),
        "BE":   np.array([1.19, 1.23, 1.17, 1.14, 1.12, 1.09, 1.07, 1.07, 1.07]),
        "BS":   np.array([1.01, 1.08, 1.09, 1.10, 1.11, 1.11, 1.10, 1.09, 1.09]),
        "BR":   np.array([1.15, 1.20, 1.16, 1.15, 1.14, 1.12, 1.09, 1.09, 1.09]),
        "BES":  np.array([0.98, 1.05, 1.07, 1.08, 1.07, 1.08, 1.07, 1.07, 1.06]),
        "BSR":  np.array([1.01, 1.06, 1.08, 1.08, 1.10, 1.09, 1.09, 1.09, 1.09]),
        "BER":  np.array([1.16, 1.19, 1.15, 1.12, 1.10, 1.09, 1.07, 1.07, 1.07]),
        "BESR": np.array([0.97, 1.02, 1.05, 1.06, 1.06, 1.06, 1.07, 1.06, 1.06]),
    },
    "Mandarin": {
        "B":    np.array([0.00, 0.00, 0.00, 1.05, 0.99, 0.93, 0.97, 0.99, 0.99]),
        "BE":   np.array([0.00, 0.00, 0.00, 1.05, 0.93, 0.91, 0.92, 0.93, 0.91]),
        "BS":   np.array([0.00, 0.00, 0.00, 1.12, 1.02, 0.95, 0.95, 0.95, 0.94]),
        "BR":   np.array([0.00, 0.00, 0.00, 1.05, 0.95, 0.90, 0.93, 0.93, 0.92]),
        "BES":  np.array([0.00, 0.00, 0.00, 1.05, 0.92, 0.92, 0.93, 0.93, 0.92]),
        "BSR":  np.array([0.00, 0.00, 0.00, 1.09, 1.00, 0.93, 0.94, 0.94, 0.93]),
        "BER":  np.array([0.00, 0.00, 0.00, 1.02, 0.88, 0.88, 0.91, 0.91, 0.90]),
        "BESR": np.array([0.00, 0.00, 0.00, 0.99, 0.89, 0.90, 0.92, 0.92, 0.91]),
    },
}

# Colors for each configuration
colors = {
    "B":    "black",
    "BE":   "dodgerblue",
    "BS":   "gold",
    "BR":   "red",
    # Mixed tones for combined configurations
    "BES":  "limegreen",
    "BSR":  "darkorange",
    "BER":  "darkviolet",
    "BESR": "brown",
}

# Plot
fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

for i, (ax, (language, configs)) in enumerate(zip(axes, rmse.items())):
    for config, values in configs.items():
        x_values = mandarin_window_centers if language == "Mandarin" else window_centers
        y_values = values[3:] if language == "Mandarin" else values
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            label=config,
            color=colors[config]
        )
    ax.set_title(language, fontsize=12)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(window_centers)

    # Only show legend in Mandarin subplot
    if language == "Mandarin":
        ax.legend(
            fontsize=9,
            title="Configuration",
            title_fontsize=9,
            loc="upper left"
        )

# Shared x axis label
axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=11)
axes[-1].set_xticklabels([f"{c:.1f}" for c in window_centers])

fig.suptitle("LCP Performance by Levenshtein Similarity Window", fontsize=13)

plt.tight_layout()
plt.savefig("levenshtein_window_analysis.png", dpi=150, bbox_inches="tight")
plt.show()