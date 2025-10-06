import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("ml_course_scores.csv")
print(df.head())

X = df[['study_hours', 'attendance', 'assignments', 'cgpa', 'project']]
y=df['score']
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train , y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R² Score:", r2)
print("Mean Squared Error:", mse)

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.show()


import numpy as np

study_range = np.arange(5, 41, 1)
pred_data = pd.DataFrame({
    'study_hours': study_range,
    'attendance': 95,
    'assignments': 90,
    'cgpa': 8.5,
    'project': 85
})

pred_scores = model.predict(pred_data)

hours_needed = study_range[np.argmin(abs(pred_scores - 90))]
print(f"To score 90%, you need to study approximately {hours_needed} hours/week.")
