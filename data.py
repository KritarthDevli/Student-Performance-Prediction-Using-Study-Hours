import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data=pd.read_excel("Study.xlsx")
X=data[["Study_Hours"]]
y=data["Final_Score"]
model=LinearRegression()
model.fit(X,y)
predicted_score=model.predict(X)
mae=mean_squared_error(y,predicted_score)
mse=mean_squared_error(y,predicted_score)
rmse=np.sqrt(mse)
r2=r2_score(y,predicted_score)
print(f"Mean absolute error is {round(mse)}, Mean squared error is {round(mse)} and root mean squarred error is {round(rmse)}, Rsqr score(Model score) is {round(r2)}")

plt.figure(figsize=(10,6))
plt.hist(data["Final_Score"],bins=30,color="skyblue",edgecolor="black")
plt.title("Distribution of final exam scores")
plt.xlabel("final exam score")
plt.ylabel("number of student")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(X,y,color="skyblue",label="Actual_Scores")
plt.plot(X,predicted_score,color="red",label="Predicted Scores (Regression Line)")
plt.title("Model prediction vs Actual score")
plt.xlabel("study hours per week")
plt.ylabel("Final output")
plt.grid(True)
plt.show()

new_hour=9
predicted_new_score=model.predict([[new_hour]])
print(f"predicted new score for {new_hour} hour  is {predicted_new_score}")




