from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from db_init import Base

class Transactions(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    FLAG = Column(Float, name="FLAG")
    avg_min_between_sent_tnx = Column(Float, name="Avg min between sent tnx")
    avg_min_between_received_tnx = Column(Float, name="Avg min between received tnx")
    time_diff_mins = Column(Float, name="Time Diff between first and last (Mins)")
    sent_tnx = Column(Float, name="Sent tnx")
    received_tnx = Column(Float, name="Received Tnx")
    number_of_created_contracts = Column(Float, name="Number of Created Contracts")
    max_value_received = Column(Float, name="max value received")
    avg_val_received = Column(Float, name="avg val received")
    avg_val_sent = Column(Float, name="avg val sent")
    total_ether_sent = Column(Float, name="total Ether sent")
    total_ether_balance = Column(Float, name="total ether balance")
    erc20_total_ether_received = Column(Float, name="ERC20 total Ether received")
    erc20_total_ether_sent = Column(Float, name="ERC20 total ether sent")
    erc20_total_ether_sent_contract = Column(Float, name="ERC20 total Ether sent contract")
    erc20_uniq_sent_addr = Column(Float, name="ERC20 uniq sent addr")
    erc20_uniq_rec_token_name = Column(Float, name="ERC20 uniq rec token name")
    erc20_most_sent_token_type = Column(String, name="ERC20 most sent token type")
    erc20_most_rec_token_type = Column(String, name="ERC20_most_rec_token_type")
    time = Column(DateTime, name="time", default=datetime.utcnow, nullable=False)

class MildlyUnsafeTransaction(Base):
    __tablename__ = "mildly_unsafe_transactions"

    uniq_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    acc_holder = Column(String)
    time = Column(DateTime, default=datetime.utcnow)
