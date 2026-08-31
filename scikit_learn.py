from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

df = pd.read_csv('concrete_data.csv')

X = df.drop(columns='concrete_compressive_strength')
y = df['concrete_compressive_strength']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = SVR(C=10, kernel='linear')
model.fit(x_train_scaled, y_train)
predictions = model.predict(x_test_scaled)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
R2_score = r2_score(y_test, predictions)

print('MAE:', mae)
print('MSE:', mse)
print('R2_SCORE:', R2_score)

