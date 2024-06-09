import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Title of the app
st.title("Analisis Asuransi")

# Description
st.write("Ini adalah aplikasi analisis data asuransi kesehatan.")
st.write("Source : https://www.youtube.com/watch?v=ntBa7YKc9XM&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=11")

# Load the dataset
df = pd.read_csv('insurance.csv')

# Display the dataframe
st.write("### Dataframe:")
st.write(df)

# Display first 5 rows of the dataframe
st.write("### 5 Baris Pertama dari Data")
st.write(df.head())

# Display the shape of the dataframe
st.write("### Bentuk Dataframe")
st.write(df.shape)

# Display info of the dataframe
st.write("### DataFrame Info")
buffer = df.info(buf=None)
st.text(buffer)

# Checking for missing values
st.write("### Nilai yang Hilang")
st.write(df.isnull().sum())

# Display statistical measures of the dataset
st.write("### Pengukuran Statistikal")
st.write(df.describe())

# Distribution of Age
st.write("### Distribusi Umur")
fig, ax = plt.subplots(figsize=(6,6))
sns.histplot(df['age'], kde=True, ax=ax)
ax.set_title('Distribusi Umur')
st.pyplot(fig)

# Gender Column
st.write("### Distribusi Jenis Kelamin")
fig, ax = plt.subplots(figsize=(6,6))
sns.countplot(x='sex', data=df, ax=ax)
ax.set_title('Distribusi Jenis Kelamin')
st.pyplot(fig)

# Display count of sex
st.write("### Jumlah Orang Menurut Jenis Kelamin")
st.write(df['sex'].value_counts())

# BMI Distribution
st.write("### Distribusi BMI")
fig, ax = plt.subplots(figsize=(6,6))
sns.histplot(df['bmi'], kde=True, ax=ax)
ax.set_title('Distribusi BMI')
st.pyplot(fig)

# Children Column
st.write("### Distribusi Anak")
fig, ax = plt.subplots(figsize=(6,6))
sns.countplot(x='children', data=df, ax=ax)
ax.set_title('Distribusi Anak')
st.pyplot(fig)

# Display count of children
st.write("### Children Count")
st.write(df['children'].value_counts())

# Smoker Column
st.write("### Distribusi Perokok")
fig, ax = plt.subplots(figsize=(6,6))
sns.countplot(x='smoker', data=df, ax=ax)
ax.set_title('Perokok')
st.pyplot(fig)

# Display count of smoker
st.write("### Jumlah Perokok")
st.write(df['smoker'].value_counts())

# Region Column
st.write("### Distribusi Wilayah")
fig, ax = plt.subplots(figsize=(6,6))
sns.countplot(x='region', data=df, ax=ax)
ax.set_title('Wilayah')
st.pyplot(fig)

# Display count of region
st.write("### Region Count")
st.write(df['region'].value_counts())

# Distribution of charges value
st.write("### Charges Distribution")
fig, ax = plt.subplots(figsize=(6,6))
sns.histplot(df['charges'], kde=True, ax=ax)
ax.set_title('Charges Distribution')
st.pyplot(fig)

# Encoding categorical columns
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Features and target variable
x = df.drop(columns='charges', axis=1)
y = df['charges']

# Display the features and target
st.write("### Features (X):")
st.write(x)

st.write("### Target (y):")
st.write(y)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Display shape of datasets
st.write("### Bentuk Set Data")
st.write(f"X: {x.shape}")
st.write(f"X_train: {x_train.shape}")
st.write(f"X_test: {x_test.shape}")

# Loading the Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Prediction on training data
training_data_prediction = regressor.predict(x_train)

# R Squared value for training data
r2_train = metrics.r2_score(y_train, training_data_prediction)
st.write("### R Squared value for Training data: ", r2_train)

# Prediction on test data
test_data_prediction = regressor.predict(x_test)

# R Squared value for test data
r2_test = metrics.r2_score(y_test, test_data_prediction)
st.write("### R Squared value for Test data: ", r2_test)

# Predicting for a single input
st.write("### Prediksi Jumlah Asuransi (dalam US Dollar)")
age = st.number_input('Umur', min_value=0, max_value=100, value=31)
sex = st.selectbox('Kelamin (0: Male, 1: Female)', [0, 1], index=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.74)
children = st.number_input('Jumlah Anak', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Perokok (0: Yes, 1: No)', [0, 1], index=1)
region = st.selectbox('Wilayah (0: Southeast, 1: Southwest, 2: Northeast, 3: Northwest)', [0, 1, 2, 3], index=0)

input_data = (age, sex, bmi, children, smoker, region)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Predicting endurance cost
prediction = regressor.predict(input_data_reshaped)
st.write(f'Prediksi Jumlah Asuransi dalam USD {prediction[0]:.2f}')