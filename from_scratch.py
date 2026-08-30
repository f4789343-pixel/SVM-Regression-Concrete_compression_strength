import pandas as pd
import numpy as np

df = pd.read_csv('concrete_data.csv')

print(df.columns)
print(df.info())

X = df.drop(columns='concrete_compressive_strength')
y = df["concrete_compressive_strength"]

print(X.shape)
print(y.shape)

indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

mean = np.mean(x_train, axis=1)
std = np.std(x_train, axis=1)

x_train_scaled = 
