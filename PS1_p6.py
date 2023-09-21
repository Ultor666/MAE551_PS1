import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from the CSV file
data = pd.read_csv(r"D:\Documents\US Documents\Fall 2023\MAE 551 Machine learning\HW1\HW1_P6_candy-data.csv")

# Define the label based on winpercent
data['label'] = data['winpercent'] > 50.0  # Popular if winpercent > 50, else not popular

# Split the data into features (sugarpercent and pricepercent) and labels (popular)
X = data[['sugarpercent', 'pricepercent']]
y = data['label']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)