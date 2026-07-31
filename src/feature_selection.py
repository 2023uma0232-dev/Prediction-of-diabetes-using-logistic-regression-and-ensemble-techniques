import pandas as pd

def add_clinical_flags(df):
    """
    Adds five binary clinical flags (NF1-NF5) based on Age, Glucose, pregnancies, BP, and BMI.
    """
    d = df.copy()
    d['NF1'] = ((d['Age'] <= 30) & (d['Glucose'] <= 140)).astype(int)           # young + normal glucose
    d['NF2'] = (d['BMI'] <= 30).astype(int)                                      # healthy BMI
    d['NF3'] = ((d['Age'] <= 30) & (d['Pregnancies'] <= 3)).astype(int)          # young + low pregnancies
    d['NF4'] = ((d['Glucose'] <= 140) & (d['BloodPressure'] <= 80)).astype(int)  # normal glucose + BP
    d['NF5'] = ((d['Glucose'] <= 140) & (d['BMI'] <= 45)).astype(int)            # normal glucose + BMI
    return d
