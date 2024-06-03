# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
dataset = pd.read_csv("Housing.csv")

# Clean and preprocess the dataset
x = dataset.iloc[:, 1:].values
y = dataset['price'].values / 1000  # price in thousands

'''
no : 0
yes : 1
furnished : 0
semi-furnished : 1
unfurnished : 2
'''
le = LabelEncoder()
x[:, 4] = le.fit_transform(x[:, 4])
x[:, 5] = le.fit_transform(x[:, 5])
x[:, 6] = le.fit_transform(x[:, 6])
x[:, 7] = le.fit_transform(x[:, 7])
x[:, 8] = le.fit_transform(x[:, 8])
x[:, 10] = le.fit_transform(x[:, 10])
x[:, 11] = le.fit_transform(x[:, 11])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

# Standardize the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Build the ANN model with dropout and regularization
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))  # Linear activation for regression

# Compile the model
ann.compile(optimizer='adam', loss='mean_squared_error')  # Mean squared error loss for regression

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model with early stopping
history = ann.fit(x_train, y_train, batch_size=32, epochs=1000, validation_split=0.2, callbacks=[early_stopping])

# Predict the test set results
y_pred = ann.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices (in thousands)')
plt.ylabel('Predicted Prices (in thousands)')
plt.title('Actual vs Predicted Prices')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

'''
Prediction

actual price : 3640000
area :	3000
bedrooms : 	2
bathrooms :	1
stories : 2	
mainroad : yes
guestroom :	no
basement : no
hotwaterheating : no	
airconditioning : yes	
parking : 0	
prefarea : no
furnishingstatus : furnished
'''
new_data = sc.transform([[3000,2,1,2,1,0,0,0,1,0,0,0]])
new_pred = ann.predict(new_data)
print("Actual Price : 3640\n Predicted Price :",new_pred)