import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import os


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    #TRAINING PHASE
    # 1 loading dataset
    housing = pd.read_csv("housing.csv")

    # 2 create straified test,train split

    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index,test_index in split.split(housing,housing["income_cat"]):
        housing.iloc[test_index].drop("income_cat", axis=1).to_csv("input.csv")
        housing = housing.iloc[train_index].drop("income_cat",axis=1)



    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attr = housing_features.select_dtypes(include=np.number).columns
    cat_attr = housing_features[["ocean_proximity"]].columns

    pipeline = build_pipeline(num_attr, cat_attr)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model trained and saved.")


else :
    #INFERENCE PHASE
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)


    data = pd.read_csv("housing.csv")
    transformed_data = pipeline.transform(data)
    predictions = model.predict(transformed_data)

    data["median_house_value"] = predictions
    data.to_csv("output.csv",index = False)
    print("Inference complete. Results saved to output.csv")




