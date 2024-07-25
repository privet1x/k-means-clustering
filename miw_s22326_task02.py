"""
This module implements the K-means clustering algorithm to assign points from a file to clusters.
The script reads point data from a file, performs K-means clustering,
and outputs the number of clusters and centroids to an output file.
It also generates a visualization of the clusters.
"""
# pylint: disable=no-member

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def read_points(file_path):
    """
    Reads points from the specified input file using numpy genfromtxt function.
    Handles irregular amounts of spaces between numbers.
    """
    return np.genfromtxt(file_path)


def initialize_centroids(points, number_clusters):
    """
    Randomly selects number_clusters centroids from the set of points.
    """
    indices = np.random.choice(points.shape[0], size=number_clusters, replace=False)
    return points[indices]


def assign_points_to_clusters(points, centroids):
    """
    Assigns each point to the nearest centroid and returns the assignments.
    Computes Euclidean distance from each point to each centroid.
    """
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def update_centroids(points, assignments, number_clusters):
    """
    Recalculates centroids based on current assignments.
    Each new centroid is the mean of points assigned to that cluster.
    """
    new_cntrds = np.array([points[assignments == i].mean(axis=0) for i in range(number_clusters)])
    return new_cntrds


def write_output(file_path, centroids):
    """
    Writes the number of clusters and centroids to the output file.
    The format includes the number of clusters followed by each centroid's coordinates.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(centroids)}\n")
        for i, centroid in enumerate(centroids):
            f.write(f"{i} {centroid[0]} {centroid[1]}\n")


def plot_clusters(points, assignments, centroids, image_path):
    """
    Plots the clusters with different colors and saves the visualization to the specified path.
    Points are colored based on their cluster assignment.
    """
    plt.figure(figsize=(8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i, color in enumerate(colors):
        plt.scatter(
            points[assignments == i, 0],
            points[assignments == i, 1],
            color=color,
            alpha=0.5,
            label=f'Cluster {i}'
        )
        plt.scatter(
            centroids[i, 0],
            centroids[i, 1],
            color=color,
            marker='x',
            s=100
        )
    plt.title('Cluster Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.savefig(image_path)
    plt.show()


def k_means(data_file_path, out_file_path, plot_file_path, number_clusters):
    """
    Main function to perform K-means clustering.
    """
    points = read_points(data_file_path)
    centroids = initialize_centroids(points, number_clusters)
    for _ in range(100):  # example max iterations
        assignments = assign_points_to_clusters(points, centroids)
        new_centroids = update_centroids(points, assignments, number_clusters)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    write_output(out_file_path, centroids)
    plot_clusters(points, assignments, centroids, plot_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: script.py <data_path> <out_path> <plot_path> <num_clusters>")
        sys.exit(1)
    _, data_path, out_path, plot_path, num_clusters = sys.argv
    k_means(data_path, out_path, plot_path, int(num_clusters))
