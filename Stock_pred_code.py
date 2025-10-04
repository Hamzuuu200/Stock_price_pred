import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score
import pickle
from sklearn.metrics import r2_score, mean_absolute_error

df=pd.read_csv('C:/Users/Ch. Hamza/Downloads/indexData.csv.zip')
print(df.head())
print(df.isnull().sum())
columns=['Open','High','Low','Close','Adj Close','Volume']
df_cleaned=df.dropna(subset=columns)
print(df_cleaned.isnull().sum())
a = df_cleaned.drop(columns=['Index'])
a['Date'] = pd.to_datetime(a['Date'])
a['Year'] = a['Date'].dt.year
a['Month'] = a['Date'].dt.month
a['Day'] = a['Date'].dt.day
a['DayOfWeek'] = a['Date'].dt.dayofweek
 = a.drop(columns=["Close", "Date"])

y = a['Close']                  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-validation R² scores:", scores)
print("Average R²:", scores.mean())
y_pred = model.predict(X_test)
print("R² score on test set:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
pickle.dump(model, open("model1.pkl", "wb"))
pickle.dump(list(X.columns), open("columns.pkl", "wb"))
pickle.dump(a, open("data.pkl", "wb"))   # <-- a still has Date
