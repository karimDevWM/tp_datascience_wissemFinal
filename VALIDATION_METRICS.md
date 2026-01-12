# Clustering Validation Metrics

Clustering is an unsupervised learning task, meaning we don't have "true" labels to check against. Therefore, we use internal validation metrics to evaluate how well the data is grouped.

## 1. Silhouette Score

The **Silhouette Score** measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).

*   **Range**: $[-1, 1]$
*   **Interpretation**:
    *   **+1**: The sample is far away from the neighboring clusters (Good separation).
    *   **0**: The sample is on or very close to the decision boundary between two neighboring clusters.
    *   **-1**: The sample might have been assigned to the wrong cluster.

A higher average silhouette score generally indicates better-defined clusters.

## 2. Davies-Bouldin Score

The **Davies-Bouldin Score** measures the average "similarity" between clusters, where similarity is a ratio of within-cluster distances to between-cluster distances.

*   **Range**: $[0, \infty)$
*   **Interpretation**:
    *   **Lower is better**.
    *   Zero is the lowest possible score.

Values closer to 0 indicate better clustering (clusters are compact and well-separated).

## Comparison

| Metric | Goal | Interpretation | Computational Cost |
| :--- | :--- | :--- | :--- |
| **Silhouette** | Maximize | $+1$ is best, $-1$ is worst | High (calculates all pairwise distances) |
| **Davies-Bouldin** | Minimize | $0$ is best | Low to Medium |

**Note**: For large datasets, calculating the Silhouette Score can be very slow. It is common to calculate it on a random sample of the data.
