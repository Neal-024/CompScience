# Sample data
data = [
    (["Apple", "High", "iOS", "5G"], "Popular"),
    (["Samsung", "Medium", "Android", "5G"], "Popular"),
    (["Google", "Medium", "Android", "No 5G"], "Not Popular"),
    (["Apple", "High", "iOS", "No 5G"], "Popular"),
    (["Samsung", "Low", "Android", "5G"], "Not Popular"),
    (["Google", "High", "Android", "5G"], "Popular"),
    (["Apple", "Medium", "iOS", "5G"], "Popular"),
    (["Samsung", "Medium", "Android", "No 5G"], "Not Popular"),
    (["Google", "Low", "Android", "5G"], "Popular"),
    (["Apple", "High", "iOS", "5G"], "Popular"),
    (["Google", "Medium", "Android", "5G"], "Popular")
]

# Calculate Prior Probabilities (how likely each class is)
class_counts = {"Popular": 0, "Not Popular": 0}

total_items = len(data)  # Total number of gadgets in the dataset

# Count occurrences of each label
for _, label in data:
    class_counts[label] += 1

# Compute priors (probability of each class)
priors = {label: count / total_items for label, count in class_counts.items()}

# Calculate Feature Probabilities (P(feature | class))
feature_counts = {"Popular": {}, "Not Popular": {}}
total_features = {"Popular": 0, "Not Popular": 0}

# Count feature occurrences per class
for features, label in data:
    for feature in features:
        if feature not in feature_counts[label]:
            feature_counts[label][feature] = 0
        feature_counts[label][feature] += 1
        total_features[label] += 1

# Prediction Function using Bayes' Theorem

def predict(features):
    probabilities = {}

    for label in class_counts:
        # Start with prior probability
        prob = priors[label]

        # Multiply by feature likelihoods (using Laplace smoothing)
        for feature in features:
            feature_count = feature_counts[label].get(feature, 0)
            numerator = feature_count + 1  # Add 1 for Laplace smoothing
            denominator = total_features[label] + len(feature_counts[label])
            feature_probability = numerator / denominator
            prob *= feature_probability

        probabilities[label] = prob

    # Print probabilities for better understanding
    print("\nProbability calculations:")
    for label, probability in probabilities.items():
        print(f"{label}: {probability:.6f}")
    
    # Return the class with the highest probability
    return max(probabilities, key=probabilities.get)

# Get User Input and Predict Popularity
def test_inputs():
    print("\nEnter gadget details:")
    brand = input("Brand (Apple, Samsung, Google): ")
    price = input("Price (High, Medium, Low): ")
    os = input("OS (iOS, Android): ")
    connectivity = input("Connectivity (5G, No 5G): ")

    user_features = [brand, price, os, connectivity]
    prediction = predict(user_features)
    print(f"\nPrediction: {prediction}.")

test_inputs()
