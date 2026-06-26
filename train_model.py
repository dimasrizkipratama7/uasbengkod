import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

print("⏳ Memulai proses pembuatan model Machine Learning...")

# 1. Load Dataset
try:
    df = pd.read_csv("Sales - Marketing customer dataset.csv")
    print("✅ Dataset 'sales_and_marketing_dataset.csv' berhasil dimuat!")
except FileNotFoundError:
    print("❌ ERROR: File 'Sales - Marketing customer dataset.csv' tidak ditemukan! Pastikan file tersebut ada di folder yang sama.")
    exit()

# 2. Data Cleaning
df = df.drop_duplicates()
kolom_drop = ['customer_id', 'signup_date', 'last_purchase_date', 'coupon_code']
df_prep = df.drop(columns=[col for col in kolom_drop if col in df.columns])

# 3. Separasi Fitur dan Target
X = df_prep.drop(columns=['churn'])
y = df_prep['churn']

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 4. Preprocessing Pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Transformasi Data
print("⚙️ Melakukan preprocessing data...")
X_preprocessed = preprocessor.fit_transform(X)

cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_features_name = numerical_features + encoded_cat_features

X_prep_df = pd.DataFrame(X_preprocessed, columns=all_features_name)

# 6. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X_prep_df, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 7. Melatih Model
print("🧠 Melatih AI (Random Forest)... Mohon tunggu sekitar 10-30 detik...")
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)

best_model = grid_rf.best_estimator_
print("✅ Model AI berhasil dilatih!")

# 8. Ekspor ke .pkl
print("💾 Menyimpan model ke dalam file .pkl...")
joblib.dump(best_model, 'model_churn_terbaik.pkl')
joblib.dump(preprocessor, 'preprocessor_churn.pkl')
joblib.dump(scaler, 'scaler_churn.pkl')
joblib.dump(X_train.columns.tolist(), 'X_columns.pkl')

print("🎉 BERHASIL! 4 File .pkl sudah terbuat di folder Anda.")
print("🚀 Sekarang Anda bisa menjalankan perintah: streamlit run app.py")