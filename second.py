import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

st.set_page_config(page_title=" LOAN APPROVAL PREDICTION", layout="wide") 
st.title(" LOAN APPROVAL PREDICTION")
st.caption("Machine learning classification using Loan data set")
@st.cache_data
def load_data(csv_path:str) ->pd.DataFrame:
    return pd.read_csv(csv_path)   
@st.cache_resource
def train_model(df:pd.DataFrame):
    target="approved"
    drop_cols=[target]

    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")


    x=df.drop(columns=drop_cols)
    y=df[target]        
    cat_cols=[c for c in ["gender","city","employment_type","bank"] if c in x.columns]
    num_cols=[c for c in x.columns if c not in cat_cols]
    numeric_transformer=Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    categorical_transformer=Pipeline(steps=[  
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("scaler",OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor=ColumnTransformer(
        transformers=[
            ("num",numeric_transformer,num_cols),
            ("cat",categorical_transformer,cat_cols)
        ])
    
    model=LogisticRegression(max_iter=2000)
    clf=Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model",model)
    ])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)

    metrics={
        "accuracy":float(accuracy_score(y_test,y_pred)),
        "precision":float(precision_score(y_test,y_pred,zero_division=0)),
        "recall":float(recall_score(y_test,y_pred,zero_division=0)),
        "f1_score":float(f1_score(y_test,y_pred,zero_division=0)),
        "confusion_matrix":confusion_matrix(y_test,y_pred).tolist()
    }                        
    return clf,metrics, x.columns.tolist()
st.sidebar.header("(1) Load Data set")
csv_path=st.sidebar.text_input(
    "CSV file path",
    value="loan_dataset.csv",
    help="put the path to the dataset csv file here"
)
try:
    df=load_data(csv_path)
except:
    st.error("Could not load the data. Please check the file path.")
    st.stop()      
st.sidebar.success(f"Data set loaded successfully.{len(df):,} records found.")    
st.sidebar.header("(2) Train Model")
train_now =st.sidebar.button("Train Model / Retrain Model")
if train_now:
    st.cache_data.clear()
clf,metrics, feature_order=train_model(df)    
colA,colB=st.columns([1,3])
with colA:
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True) 
with colB:
    st.subheader("Model Performance Metrics")
    st.write({
        "Accuracy":round(metrics['accuracy'],4),
        "Precision":round(metrics['precision'],4),
        "Recall":round(metrics['recall'],4),
        "F1_score":round(metrics['f1_score'],4),
    })
    cm=np.array(metrics['confusion_matrix'])
    st.write("Confusion Matrix (rows: Actual[0.1], columns: Predicted [0,1])")
    st.dataframe(
        pd.DataFrame(cm,columns=["Predicted 0","Predicted 1"],index=["Actual 0","Actual 1"]),
        use_container_width=True
    )
    st.divider()          

    st.subheader("Make a Prediction")
    c1,c2,c3,c4=st.columns(4)
    with c1:
        applicant_name=st.text_input("Applicant Name",value="Muhammad Ali")
        gender=st.selectbox("Gender",["Male","Female"],index=0)
        age=st.slider("Age",18,70,30)
    with c2:
         city=st.selectbox("City",sorted(df["city"].unique().tolist()))
         employment_type=st.selectbox("Employment Type",sorted(df["employment_type"].unique().tolist()))
         bank=st.selectbox("Bank",sorted(df["bank"].unique().tolist()))    
    with c3:
         income=st.number_input("Income",min_value=1500,max_value=500000, value=12000, step=1000)
         Credit_score=st.slider("Credit Score",300, 900 ,600)

    with c4:
           loan_amount_pkr=st.number_input("Loan Amount (PKR)",min_value=50000, max_value=5000000, value=250000, step=5000)
           loan_tenure_months=st.selectbox("Loan Tenure (Months)",[6,12,18,24,36,48,60], index=2)
           existing_loans=st.selectbox("Existing Loan",[0,1,2,3], index=0)
           default_history=st.selectbox("Default History",[0,1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1) ", index=0)
           has_credit_card=st.selectbox("Has Credit Card",[0,1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1) ", index=1)
    input_rows=pd.DataFrame([{
        "gender":gender,
        "age":age,
        "city":city,
        "employment_type":employment_type,
        "bank":bank,
        "monthly_income_pkr":income,
        "credit_score":Credit_score,
        "loan_amount_pkr":loan_amount_pkr,
        "loan_tenure_months":loan_tenure_months,
        "existing_loans":existing_loans,
        "default_history":default_history,
        "has_credit_card":has_credit_card
            }])
    input_rows=input_rows[feature_order]
    if st.button("Predict Loan Approval"):
        prob=clf.predict_proba(input_rows)[:,1][0]
        pred=int(prob>=0.5)
        if pred==1:
            st.success(f"Loan Approved with probability of {prob:.2%}")
        else:
            st.error(f"Loan Not Approved with probability of {1-prob:.2%}") 