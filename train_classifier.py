import pickle

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open(r"C:\Users\srini\Desktop\sign\sign-language-detector-python\data.pickle", 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

# Print accuracy
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model_xgboost.p', 'wb') as f:
    pickle.dump({'model': model}, f)
