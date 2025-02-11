
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

def initialize_centroids(data, k):
    centroids = []
    for _ in range(k):
        centroid = data.pop(0) 
        centroids.append(centroid)
    return centroids

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_index = distances.index(min(distances))
        clusters[closest_centroid_index].append(point)
    return clusters

def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            mean = [sum(coords) / len(cluster) for coords in zip(*cluster)]
            new_centroids.append(mean)
    return new_centroids

def kmeans(data, k, max_iterations=100):
    # Initialize centroids
    centroids = initialize_centroids(data.copy(), k)
    
    for _ in range(max_iterations):
        # Assign clusters
        clusters = assign_clusters(data, centroids)
        
        # Update centroids
        new_centroids = update_centroids(clusters)
        
        # Check for convergence
        if new_centroids == centroids:
            break
        
        centroids = new_centroids
    
    return clusters, centroids

# Example usage
if __name__ == "__main__":
    # Sample data: 2D points
    data = [
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 8],
        [7, 8],
        [8, 9]
    ]
    
    # Number of clusters
    k = 2
    
    # Run K-Means
    clusters, centroids = kmeans(data, k)
    
    # Print results
    print("Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
    print("\nCentroids:")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")