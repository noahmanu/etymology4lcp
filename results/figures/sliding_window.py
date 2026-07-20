import numpy as np
import matplotlib.pyplot as plt

# Define window centers (normalized Levenshtein distance)
window_centers = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Dummy RMSE values per language and configuration
# Mandarin: word relatedness features should help less due to different script/language family
rmse = {
    "German": {
        "B":    np.array([0.45, 0.43, 0.42, 0.44, 0.46, 0.48, 0.47, 0.45, 0.44]),
        "BE":   np.array([0.43, 0.41, 0.40, 0.41, 0.43, 0.45, 0.44, 0.43, 0.42]),
        "BS":   np.array([0.43, 0.41, 0.39, 0.40, 0.42, 0.44, 0.43, 0.42, 0.41]),
        "BR":   np.array([0.44, 0.41, 0.38, 0.36, 0.37, 0.41, 0.44, 0.43, 0.43]),
        "BSR":  np.array([0.43, 0.40, 0.37, 0.34, 0.35, 0.39, 0.43, 0.42, 0.42]),
        "BER":  np.array([0.42, 0.39, 0.36, 0.33, 0.34, 0.38, 0.42, 0.41, 0.41]),
        "BESR": np.array([0.41, 0.38, 0.35, 0.31, 0.32, 0.36, 0.40, 0.40, 0.40]),
    },
    "Spanish": {
        "B":    np.array([0.46, 0.44, 0.43, 0.45, 0.47, 0.49, 0.48, 0.46, 0.45]),
        "BE":   np.array([0.44, 0.42, 0.41, 0.42, 0.44, 0.46, 0.45, 0.44, 0.43]),
        "BS":   np.array([0.44, 0.42, 0.40, 0.41, 0.43, 0.45, 0.44, 0.43, 0.42]),
        "BR":   np.array([0.45, 0.42, 0.39, 0.36, 0.37, 0.41, 0.45, 0.44, 0.44]),
        "BSR":  np.array([0.44, 0.41, 0.37, 0.34, 0.35, 0.39, 0.43, 0.43, 0.43]),
        "BER":  np.array([0.43, 0.40, 0.36, 0.33, 0.34, 0.38, 0.42, 0.42, 0.42]),
        "BESR": np.array([0.42, 0.39, 0.35, 0.31, 0.32, 0.36, 0.41, 0.41, 0.41]),
    },
    "Mandarin": {
        "B":    np.array([0.47, 0.46, 0.46, 0.47, 0.47, 0.48, 0.47, 0.46, 0.46]),
        "BE":   np.array([0.45, 0.44, 0.43, 0.45, 0.45, 0.46, 0.45, 0.44, 0.44]),
        "BS":   np.array([0.46, 0.45, 0.44, 0.46, 0.46, 0.47, 0.46, 0.45, 0.45]),
        "BR":   np.array([0.46, 0.45, 0.45, 0.46, 0.46, 0.47, 0.46, 0.45, 0.46]),
        "BSR":  np.array([0.45, 0.44, 0.44, 0.45, 0.45, 0.46, 0.45, 0.44, 0.45]),
        "BER":  np.array([0.44, 0.43, 0.42, 0.44, 0.44, 0.45, 0.44, 0.43, 0.44]),
        "BESR": np.array([0.43, 0.42, 0.41, 0.43, 0.43, 0.44, 0.43, 0.42, 0.43]),
    },
}

# Colors for each configuration
colors = {
    "B":    "black",
    "BE":   "steelblue",
    "BS":   "mediumseagreen",
    "BR":   "darkorange",
    "BSR":  "mediumpurple",
    "BER":  "crimson",
    "BESR": "saddlebrown",
}

# Plot
fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

for i, (ax, (language, configs)) in enumerate(zip(axes, rmse.items())):
    for config, values in configs.items():
        ax.plot(
            window_centers,
            values,
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

    # Only show legend in first subplot
    if i == 0:
        ax.legend(
            fontsize=9,
            title="Configuration",
            title_fontsize=9,
            loc="upper right"
        )

# Shared x axis label
axes[-1].set_xlabel("Normalized Levenshtein Distance (Window Center)", fontsize=11)
axes[-1].set_xticklabels([f"{c:.1f}" for c in window_centers])

fig.suptitle("LCP Performance by Surface Similarity Window (Dummy Data)", fontsize=13)

plt.tight_layout()
plt.savefig("dummy_window_analysis.png", dpi=150, bbox_inches="tight")
plt.show()