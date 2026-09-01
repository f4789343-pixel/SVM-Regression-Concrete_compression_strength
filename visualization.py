import matplotlib.pyplot as plt
import numpy as np
from from_scratch import x_train_scaled, y_train

epsilon = 5
predictions = []
x = x_train_scaled[:, 0]

w = 0.0
b = 0.0
C = 100
lr = 0.01
for epoch in range(1000):

    for i in range(len(x)):

        raw_pred = w * x[i] + b
        error = y_train[i] - raw_pred

        if error > epsilon:
            dw = w - C * x[i]
            db = -C

        elif error < -epsilon:
            dw = w + C * x[i]
            db = C

        else:
            dw = w
            db = 0

        w -= lr * dw
        b -= lr * db

x_line = np.linspace(x.min(), x.max(), 200)

prediction = w * x_line + b
upper = prediction + epsilon
lower = prediction - epsilon

predictions = w * x + b
errors = y_train - predictions

outside = np.abs(errors) > epsilon

plt.scatter(x, y_train, label="Data")


plt.plot(x_line, prediction, label='SVR prediction')

plt.plot(x_line, upper, linestyle='--', label='+ epsilon')

plt.plot(x_line, lower, linestyle='--', label='- epsilon')

error = y_train - predictions

outside = np.abs(error) > epsilon

plt.scatter(x[outside], y_train[outside],s=60, facecolors='none', edgecolors='black', label='Outside epsilon tube')

plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.title("SVR: Prediction Line and Epsilon Tube")
plt.legend()
plt.grid()
plt.savefig('SVR.png')
plt.show()