import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Import the Processed data from model_dev2
df = pd.read_csv('model_dev2/data/processed/processed_leading_deaths.csv')
len(df)

# drop rows with missing values
df.dropna(inplace=True)
len(df)

# Define the features and the target variable "deaths"
X = df.drop('deaths', axis=1) 
y = df['deaths'] 

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X)
pickle.dump(scaler, open('model_dev2/models/scaler_deaths.sav', 'wb')) #The "wb" mode ensures that the file is opened for writing in binary mode.

# Fit the scaler to the features and transform
X_scaled = scaler.transform(X)

# Split the scaled data into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the size of each set
(X_train.shape, X_val.shape, X_test.shape)

# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('model_dev2/models/X_train.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('model_dev2/models/X_columns.sav', 'wb'))