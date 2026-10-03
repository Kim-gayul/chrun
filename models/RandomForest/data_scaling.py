import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path, skipinitialspace=True)
    return df

def preprocessor():
    categorical_columns = ['Gender','Contract','PaymentMethod'] 
    numeric_columns = ['Tenure','MonthlyCharges', 'TotalCharges','Age'] # CustomerID는 의미 없어서 지움

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())  
    ])

    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("category", category_pipeline, categorical_columns), 
        ("number", numeric_pipeline, numeric_columns)
    ])

    return preprocessor

def preprocess_data(df):
    x = df.drop(['CustomerID', 'Churn'],axis=1) # Churn 가 열에 있어서 axis = 1
    y = df['Churn']
    
    x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=0, stratify= y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify= y_train) # train 60%, val 20%, test 20% 비율로 생성
    
    processor = preprocessor()

    x_train = processor.fit_transform(x_train)
    x_val = processor.transform(x_val)
    x_test = processor.transform(x_test)
    
    le = LabelEncoder()       # df['Churn']값이 yes,no 라서 수치형으로 변환
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)


    return x_train, x_val, x_test, y_train, y_val, y_test
