from fastapi import FastAPI, Depends
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel


# Load model and encoder
model_data = joblib.load("hearts_failure_prediction.pkl")
model = model_data["model"]
encoder = model_data["encoder"]

apping = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./heart_model.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#  SQLAlchemy Model for Logging
class PredictionLogDB(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    Age = Column(Integer)
    Sex = Column(Integer)
    ChestPainType = Column(Integer)
    RestingBP = Column(Integer)
    Cholesterol = Column(Integer)
    FastingBS = Column(Integer)
    RestingECG = Column(Integer)
    MaxHR = Column(Integer)
    HeartDisease = Column(Integer)

Base.metadata.create_all(bind=engine)

#  Pydantic Model for API Input
class PredictionLog(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@apping.post("/predict/")
def predict(data: PredictionLog, db=Depends(get_db)):
    features = np.array([[
        data.Age, 
        data.Sex,  
        data.ChestPainType,  
        data.RestingBP, 
        data.Cholesterol, 
        data.FastingBS, 
        data.RestingECG,  
        data.MaxHR
    ]])

    prediction = model.predict(features)
    heart_disease_pred = int(prediction[0])

    #  Log prediction in the database
    log_entry = PredictionLogDB(
        Age=data.Age,
        Sex=data.Sex,
        ChestPainType=data.ChestPainType,
        RestingBP=data.RestingBP,
        Cholesterol=data.Cholesterol,
        FastingBS=data.FastingBS,
        RestingECG=data.RestingECG,
        MaxHR=data.MaxHR,
        HeartDisease=heart_disease_pred
    )
    db.add(log_entry)
    db.commit()

    return {"HeartDisease": heart_disease_pred}

@apping.get("/predictions/")
def get_predictions(db=Depends(get_db)):
    records = db.query(PredictionLogDB).all()
    return records

#Age:int, Sex: str, ChestPainType: str, RestingBP: int, Cholesterol: int, FastingBS: int, RestingECG: str, MaxHR: int,