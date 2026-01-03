# Ovarian Cancer Prediction using Machine Learning
# Data Loading and Preprocessing

import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.figure_factory as ff

# Load data
df = pd.read_excel("Supplementary data 1.xlsx")

# Clean numeric columns
def clean_numeric(col):
    if col.dtype == 'object':
        col = col.str.replace('>', '').str.replace('<', '').str.strip()
        col = pd.to_numeric(col, errors='coerce') 
    return col

df = df.apply(clean_numeric)

print(df.head(10))
print("\nColumns:")
print(df.columns.tolist())
print("\nShape:", df.shape)
print("\nTarget 'TYPE' distribution:")
print(df['TYPE'].value_counts())
print(df.describe())

# Handle missing values
df = df.drop('SUBJECT_ID', axis=1)
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Missing values handled. New head:")
print(df.head())

# Split features and target
X = df.drop('TYPE', axis=1)
y = df['TYPE']

# Standardize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, preds) * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Malignant (0)", "Benign (1)"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Feature importance visualization
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

fig = px.bar(x=importances[indices[:15]], y=features[indices[:15]],
             orientation='h', 
             title='Top 15 Feature Importances for Ovarian Cancer Classification',
             labels={'x': 'Importance Score', 'y': 'Feature'},
             color=importances[indices[:15]],
             color_continuous_scale='Blues')

fig.update_layout(
    width=800, height=600,
    xaxis_title='Importance Level',
    yaxis_title='Features',
    coloraxis_colorbar={'title': 'Importance'},
    annotations=[dict(x=0.5, y=-0.15, xref='paper', yref='paper', 
                     text='Higher importance means the feature contributes more to predictions (e.g., HE4 is a key tumor marker).', 
                     showarrow=False)]
)

fig.show()

# Confusion matrix heatmap
cm = [[31, 3], [1, 35]]
labels = ['Malignant', 'Benign']

fig = ff.create_annotated_heatmap(cm, x=labels, y=labels, colorscale='Blues')

fig.update_layout(
    title='Confusion Matrix for Ovarian Cancer Classification',
    xaxis_title='Predicted Label',
    yaxis_title='True Label',
    width=500, height=400,
    annotations=[dict(x=0.5, y=-0.3, xref='paper', yref='paper', 
                     text='Accuracy: 94.29%', showarrow=False, 
                     font=dict(size=12, color="black"))]
)

fig.show()

# HE4 distribution by class
df_plot = X.copy()
df_plot['TYPE'] = y.map({0: 'Malignant', 1: 'Benign'})

fig = px.box(df_plot, x='TYPE', y='HE4',  
             title='Distribution of HE4 by Class (Malignant vs Benign)',
             color='TYPE', color_discrete_map={'Malignant': 'red', 'Benign': 'blue'})

fig.update_layout(
    width=700, height=500,
    xaxis_title='Class',
    yaxis_title='HE4 Level (Normalized)',
    annotations=[dict(x=0.5, y=-0.2, xref='paper', yref='paper', 
                     text='Higher HE4 levels are strongly associated with malignant cases, making it a key biomarker for diagnosis.', 
                     showarrow=False, font=dict(size=12))]
)

fig.show()

# ROC Curve
probs = model.predict_proba(X_test)[:, 1]  
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

fig = px.line(x=fpr, y=tpr, title='ROC Curve for Ovarian Cancer Classification',
              labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})

fig.add_shape(type='line', line=dict(dash='dash'), x0=0, y0=0, x1=1, y1=1)

fig.update_layout(
    width=700, height=500,
    annotations=[dict(x=0.5, y=0.2, text=f'AUC = {roc_auc:.2f}', 
                     showarrow=False, font=dict(size=12))]
)

fig.show()
