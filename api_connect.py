from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import numpy as np
import joblib
import os
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
from db_init import SessionLocal, engine
from db_handling import Base, Transactions, MildlyUnsafeTransaction
from fault_reason import reason
import pandas as pd
from tensorflow.keras.models import load_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

model = load_model(os.path.join("model", "fraud_detect.keras"))
pack = joblib.load(os.path.join("model", "models.joblib"))
enc1 = pack['enc1']
enc2 = pack['enc2']
sc = pack['scaler']

class Transaction_data(BaseModel):
    acc_holder: str
    features: list[Union[float , str]]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def changes_in_dataset(label_int, data, db: Session):
    transaction = Transactions(
        FLAG=float(label_int),
        avg_min_between_sent_tnx=float(data[0]),
        avg_min_between_received_tnx=float(data[1]),
        time_diff_mins=float(data[2]),
        sent_tnx=float(data[3]),
        received_tnx=float(data[4]),
        number_of_created_contracts=float(data[5]),
        max_value_received=float(data[6]),
        avg_val_received=float(data[7]),
        avg_val_sent=float(data[8]),
        total_ether_sent=float(data[9]),
        total_ether_balance=float(data[10]),
        erc20_total_ether_received=float(data[11]),
        erc20_total_ether_sent=float(data[12]),
        erc20_total_ether_sent_contract=float(data[13]),
        erc20_uniq_sent_addr=float(data[14]),
        erc20_uniq_rec_token_name=float(data[15]),
        erc20_most_sent_token_type=str(data[16]),
        erc20_most_rec_token_type=str(data[17]),
        time=datetime.now()
    )
    db.add(transaction)
    db.commit()


def encoding(encoder, val):
    try:
        if pd.isna(val) or val == "":
            val = "missing"
        return encoder.transform([val])[0]
    except ValueError:
        return encoder.transform(['missing'])[0]

@app.get("/ping")
def ping():
    return {"message": "server is running"}

@app.post("/predict")
def predict(data: Transaction_data, db: Session = Depends(get_db)):
    try:
        acc_holder = data.acc_holder
        data1 = ["missing" if pd.isna(x) else x for x in data.features]
        data2 = data1.copy()

        if len(data2) != 18:
            return {"ML error": "Missing features. Expected 18 features."}

        data2[-2] = encoding(enc1, data2[-2])
        data2[-1] = encoding(enc2, data2[-1])

        input_data = np.array(data2).reshape(1, -1)
        input_data = sc.transform(input_data)
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        filters = [
            Transactions.FLAG == float(prediction),
            Transactions.avg_min_between_sent_tnx == float(data1[0]),
            Transactions.avg_min_between_received_tnx == float(data1[1]),
            Transactions.time_diff_mins == float(data1[2]),
            Transactions.sent_tnx == float(data1[3]),
            Transactions.received_tnx == float(data1[4]),
            Transactions.number_of_created_contracts == float(data1[5]),
            Transactions.max_value_received == float(data1[6]),
            Transactions.avg_val_received == float(data1[7]),
            Transactions.avg_val_sent == float(data1[8]),
            Transactions.total_ether_sent == float(data1[9]),
            Transactions.total_ether_balance == float(data1[10]),
            Transactions.erc20_total_ether_received == float(data1[11]),
            Transactions.erc20_total_ether_sent == float(data1[12]),
            Transactions.erc20_total_ether_sent_contract == float(data1[13]),
            Transactions.erc20_uniq_sent_addr == float(data1[14]),
            Transactions.erc20_uniq_rec_token_name == float(data1[15]),
            Transactions.erc20_most_sent_token_type == str(data1[16]),
            Transactions.erc20_most_rec_token_type == str(data1[17])
        ]

        row_exists = db.query(Transactions).filter(and_(*filters)).first() is not None

        if prediction == 1 and confidence >= 0.70:
            label = "Fraud"
            fr_type = reason(str(data1))

            if row_exists:
                print("Row is present in the database so no changes made.")
            else:
                print("Row is not present in the database so updating it.")
                changes_in_dataset(prediction, data1, db)

        elif 0.50 <= confidence < 0.70:
            label = "Non - Fraud"
            fr_type = "Mildly Unsafe Transaction"+ " - " + reason(str(data1))

            count = db.query(MildlyUnsafeTransaction).filter(MildlyUnsafeTransaction.acc_holder == acc_holder).count()
            if count > 2:
                db.query(MildlyUnsafeTransaction).filter(MildlyUnsafeTransaction.acc_holder == acc_holder).delete()
                db.commit()
                label = "Fraud"
                fr_type = "Reciever's Account found too many times in Mildly Fraud Transactions records."
                print("More than 2 suspicious records found, marked as fraud.")
            else:
                new_entry = MildlyUnsafeTransaction(acc_holder=acc_holder)
                db.add(new_entry)
                db.commit()
                print("Added to mildly unsafe monitoring table.")

        else:
            label = "Non - Fraud"
            fr_type = "Safe Transaction. No possible fraud found."

        return {
            "prediction": label,
            "Type": fr_type,
            "confidence": f"{confidence*100:.2f}%"
        }
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": "An error occurred while processing the request."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)