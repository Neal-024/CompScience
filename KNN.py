
def euclidean_distance(point_1, point_2):
    distance = 0
    for i in range(len(point_1)):
        distance += (point_1[i] - point_2[i]) ** 2  
    return distance ** 0.5  

def knn_predict(training_data, test_point, k):
    distances = []

    for data_point in training_data:
        features = data_point[:-1]  
        label = data_point[-1]       
        distance = euclidean_distance(features, test_point)
        distances.append((distance, label))  

    distances.sort(key=lambda x: x[0])  
    nearest_neighbors = distances[:k]   

    class_votes = {}
    for neighbor in nearest_neighbors:
        label = neighbor[1] 
        if label in class_votes:
            class_votes[label] += 1 
        else:
            class_votes[label] = 1

    return max(class_votes, key=class_votes.get)

if __name__ == "__main__":

    # Training data
    training_data = [
        [40, 20, "Red"],  
        [50, 50, "Blue"], 
        [60, 90, "Blue"],  
        [10, 25, "Red"],  
        [70, 70, "Blue"],  
        [60, 10, "Red"],  
        [25, 80, "Blue"]  
    ]
    
    # Test 
    brightness = int(input("Enter Brightness: "))
    saturation = int(input("Enter Saturation: "))

    test = [brightness, saturation]  
    
    # Number of neighbors 
    k = 3
    
    # Predict 
    prediction = knn_predict(training_data, test, k)
    print(f'Brightness of {brightness} and Saturation of {saturation} is classified as "{prediction}"')