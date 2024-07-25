# K-means Clustering Algorithm

This module implements the K-means clustering algorithm to assign points from a file to clusters. The script reads point data from a file, performs K-means clustering, and outputs the number of clusters and centroids to an output file. It also generates a visualization of the clusters.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Function Descriptions](#function-descriptions)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The K-means clustering algorithm partitions data points into a specified number of clusters. The algorithm iteratively refines the positions of the centroids and the assignments of the data points to the clusters. This implementation includes reading points from a file, initializing centroids, assigning points to clusters, updating centroids, and visualizing the results.

## Requirements

- Python 3.x
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

You can install the required packages using the following command:
```bash
pip install numpy matplotlib
```

## Function Descriptions

- `read_points(file_path)`: Reads points from the specified input file using NumPy's `genfromtxt` function. Handles irregular amounts of spaces between numbers.
- `initialize_centroids(points, number_clusters)`: Randomly selects a specified number of centroids from the set of points.
- `assign_points_to_clusters(points, centroids)`: Assigns each point to the nearest centroid and returns the assignments. Computes Euclidean distance from each point to each centroid.
- `update_centroids(points, assignments, number_clusters)`: Recalculates centroids based on current assignments. Each new centroid is the mean of points assigned to that cluster.
- `write_output(file_path, centroids)`: Writes the number of clusters and centroids to the output file. The format includes the number of clusters followed by each centroid's coordinates.
- `plot_clusters(points, assignments, centroids, image_path)`: Plots the clusters with different colors and saves the visualization to the specified path. Points are colored based on their cluster assignment.
- `k_means(data_file_path, out_file_path, plot_file_path, number_clusters)`: Main function to perform K-means clustering.

## File Descriptions

- `miw_s22326_task02.py`: Main script containing the K-means clustering algorithm and related functions.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
