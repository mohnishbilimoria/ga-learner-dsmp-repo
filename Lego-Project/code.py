# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
print(df.head())

X = df.drop('list_price',axis=1)
y = df['list_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 6)
# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
cols = X_train.columns
fig, axes = plt.subplots(nrows=3, ncols=3)

for i in range(3):
    for col in cols:
        #axes[1, 1].scatter(x, y)
        plt.scatter(x=df[col],y=df['list_price'], alpha=0.5)

plt.show()


# code ends here



# --------------
# Code starts here
corr = X_train.corr()
print(corr)

X_train.drop('play_star_rating', axis=1, inplace=True)
X_train.drop('val_star_rating', axis=1, inplace=True)

X_test.drop('play_star_rating', axis=1, inplace=True)
X_test.drop('val_star_rating', axis=1, inplace=True)

# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print(mse)

r2 = r2_score(y_test,y_pred)
print(r2)

# Code ends here


# --------------
# Code starts here

residual = y_test - y_pred
plt.hist(residual,bins=10)
plt.show()

# Code ends here


