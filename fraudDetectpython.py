# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:40:09 2025

@author: Peezicus
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report, confusion_matrix  

# Load dataset  
df = pd.read_csv("creditcard.csv")  

# Display basic information  
print(df.info())  
print(df.head())  

# Check fraud distribution  
print(df["Class"].value_counts())  

# Plot fraud vs. non-fraud transactions  
plt.figure(figsize=(6,4))  
sns.countplot(x="Class", data=df, palette="coolwarm")  
plt.title("Distribution of Fraudulent Transactions")  
plt.show()

# Separate features and target variable  
X = df.drop(columns=["Class"])  
y = df["Class"]  

# Standardize numerical features  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Split data into training (80%) and testing (20%)  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)  

print(f"Training data size: {X_train.shape}")  
print(f"Testing data size: {X_test.shape}")  

# Train a Random Forest model  
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X_train, y_train)  

# Make predictions  
y_pred = rf_model.predict(X_test)  

# Evaluate model performance  
print("Classification Report:\n", classification_report(y_test, y_pred))  
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  

# Visualize Confusion Matrix  
plt.figure(figsize=(5,4))  
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="coolwarm", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])  
plt.title("Confusion Matrix")  
plt.xlabel("Predicted")  
plt.ylabel("Actual")  
plt.show()
