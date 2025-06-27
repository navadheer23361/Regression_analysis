import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data_path =  'Housing.csv'
data = pd.read_csv(data_path)

info = data.info()
describe = data.describe()
missing_values = data.isnull().sum() 

print(info)
print(describe)
# print(missing_values) # no missing values in the data

# splitting the data into to test and train sets

features = ['area','bedrooms','bathrooms','stories','parking']
X = data[features]
y = data["price"]

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=1)

model = LinearRegression()
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
# print("predictions are: \n" ,y_pred)

# Evaluating model using MAE, MSE, R²

mae = mean_absolute_error(test_y, y_pred)
mse = mean_squared_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)

print("Model Evaluations :")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# plotting regression line

plt.figure(figsize=(8, 5))
plt.scatter(test_y, y_pred, color='green', edgecolor='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.show()

# Interpret and Coefficients

print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: {coef:.2f}")

