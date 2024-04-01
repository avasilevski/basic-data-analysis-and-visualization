import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def predict_species(sepal_length, sepal_width, petal_length, petal_width, model, x_train, y_train):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    predicted_species = model.predict(features)[0]
    
    neighbors = model.kneighbors(features, return_distance=False)
    neighbor_labels = y_train.iloc[neighbors[0]].values
    confidence = sum(neighbor_labels == predicted_species) / len(neighbor_labels)
    
    return predicted_species, confidence

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, names=column_names)

x = iris.drop('species', axis=1)
y = iris['species']
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)

# Input values for prediction
sepal_length = 8.0
sepal_width = 3.0
petal_length = 2.0
petal_width = 2.0

predicted_species, confidence = predict_species(sepal_length, sepal_width, petal_length, petal_width, knn, x_train, y_train)
print(f"Predicted Species: {predicted_species}, Confidence: {confidence:.2f}")
