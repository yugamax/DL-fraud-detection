import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os
from db_init import SessionLocal
from db_handling import Transactions
import joblib
import warnings

warnings.filterwarnings("ignore")

model_dir = os.path.join("model")
os.makedirs(model_dir, exist_ok=True)

session = SessionLocal()

query = session.query(Transactions)
df = pd.read_sql(query.statement, session.bind)
session.close()

df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.fillna(df.median(numeric_only=True))

df['ERC20 most sent token type'] = df['ERC20 most sent token type'].fillna('missing').astype(str)
df['ERC20_most_rec_token_type'] = df['ERC20_most_rec_token_type'].fillna('missing').astype(str)

enc1 = LabelEncoder()
df['ERC20 most sent token type'] = enc1.fit_transform(df['ERC20 most sent token type'])

enc2 = LabelEncoder()
df['ERC20_most_rec_token_type'] = enc2.fit_transform(df['ERC20_most_rec_token_type'])

x = df.iloc[:, 2:20]
y = df.iloc[:, 1]

sc = StandardScaler()
x = sc.fit_transform(x)

balancingf = y.value_counts()[0] / y.value_counts()[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, random_state=42, stratify=y)

model = XGBClassifier(scale_pos_weight=balancingf, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("\t\tClassification Report:\n")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy of model : {model.score(x_test, y_test)*100:.2f} %")

pack = {'model': model, 'enc1': enc1, 'enc2': enc2 , 'scaler': sc}
joblib.dump(pack, os.path.join(model_dir, "models.joblib"))