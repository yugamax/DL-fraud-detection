import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

sc = MinMaxScaler()
x = sc.fit_transform(x)

balancingf = y.value_counts()[0] / y.value_counts()[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, random_state=42, stratify=y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

bestmod = ModelCheckpoint("model\gasfee.keras", monitor='val_accuracy', save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=60, batch_size=16, validation_split=0.20, callbacks=[es, lr, bestmod])

pack = {'model': model, 'enc1': enc1, 'enc2': enc2 , 'scaler': sc}
joblib.dump(pack, os.path.join(model_dir, "models.joblib"))

yp = model.predict(x_test)
yp_class = np.argmax(yp, axis=1)
acc = accuracy_score(y_test, yp_class)
print(f"Model accuracy : {(acc*100):.2f} %")
print("\t\tClassification Report:\n")
print(classification_report(y_test, yp))