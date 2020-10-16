import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('taxi.csv')

print(data.head(10))

x_real = data.iloc[:, 0:-1].values #idependent variable
y_real = data.iloc[:, -1]. values #dependent variable

print(y_real)


# time to split
x_train, x_test, y_train, y_test = train_test_split(x_real,y_real, test_size = 0.3, random_state = 0)

# now we have to apply algorithm we import LinearRegression

reg = LinearRegression()

# now the model is trained
reg.fit(x_train,y_train)

print("Train Score:", reg.score(x_train, y_train))
print("Test Score:", reg.score(x_test, y_test))

pickle.dump(reg, open('taxi.pkl','wb'))

model = pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80, 1770000, 6000, 85]]))
