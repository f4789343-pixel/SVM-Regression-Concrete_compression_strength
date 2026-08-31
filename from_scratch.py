import pandas as pd
import numpy as np

df = pd.read_csv('concrete_data.csv')

print(df.columns)
print(df.info())

X = df.drop(columns='concrete_compressive_strength')
y = df["concrete_compressive_strength"]

print(X.shape)
print(y.shape)
np.random.seed(42)
indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train_scaled = (x_train - mean) / std
x_test_scaled = (x_test - mean) / std
print(x_train_scaled.shape)
w = np.zeros(len(x_train[0]))
b = 0
c = 100
lr = 0.01
epsilon = 2
print('w',w.shape)
for epoch in range(1000):
  for i in range(len(x_train_scaled)):
    raw_pred = np.dot(w,x_train_scaled[i])+b
    error = y_train[i] - raw_pred
    loss = max(0, abs(error)-epsilon)
    if error > epsilon:
       dw = w - c*x_train_scaled[i]
       db = -c
    elif error < -epsilon:
       dw = w + c*x_train_scaled[i]
       db = c
    else:
       dw = w
       db = 0
    w -= lr*dw
    b -= lr*db

errors = []
for i in range(len(x_test_scaled)):
   raw_test_pred = np.dot(w, x_test_scaled[i])+b
   error2 = y_test[i] - raw_test_pred
   errors.append(error2)

errors = np.array(errors)
mae = np.mean(np.abs(errors))
mse = np.mean((errors**2))
res = np.sum(errors**2)
total = np.sum((y_test-np.mean(y_test))**2)
r2 = 1 - (res/total)
print('mae', mae)
print('mse', mse)
print('r2', r2)





