#creating dataframe
import pandas as pd
df = pd.read_csv("/content/Fifa 23 Players Data.csv", on_bad_lines="skip")


#selecting the position
mask = df["Best Position"]=="RM"
df_mask = df[mask]
df_mask.head(5)     


y = df_mask["Overall"]     #target is player overall rating
X = df_mask.iloc[:,38:-22]     #features are all the individual attributes e.g. short passing, finishing, strength



#preprocessing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().set_output(transform='pandas')

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)



#linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_scaled, y_train)

print(model.score(X_test_scaled, y_test))

#given all the individual attributes we can predict the player's overall rating with a very high degree of accuracy



#displaying coefficients
import numpy as np

coefficients = model.coef_     #this retrieves the weights assigned to each feature in the linear regression

feature_names = X_train_scaled.columns     #fetching the names of the features so we know which attributes the weights belong to

coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})     #putting the results in a dataframe

print(coefficients_df.sort_values(by=['Coefficient'], ascending=False).reset_index(drop=True))     #displaying results

