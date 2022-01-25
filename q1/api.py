# Dependencies
import base64
import json
import io
import pandas as pd
from q1b import create_data_splits, train_model, test_model
import pickle
#  from sklearn.externals import joblib
import joblib
from flask import Flask, request
# import numpy as np

# Your API definition
app = Flask(__name__)


@app.route('/test', methods=['POST'])
def test():
    record = json.loads(request.data)
    model = pickle.loads(base64.b64decode(record["model"]))
    df = pd.read_csv(io.StringIO(record["dataset"]))
    results = test_model(df, model)
    print(results)
    return {
        "results": results
    }

"""
dataset csv encoded in the data field
returns json with train and test fields
"""
@app.route('/split', methods=['POST'])
def split():
    record = json.loads(request.data)
    dataset_string = record["data"]
    dataframe = pd.read_csv(io.StringIO(dataset_string))
    train_df, test_df = create_data_splits(dataframe, record["size"])
    train_str = io.StringIO()
    test_str = io.StringIO()
    train_df.to_csv(train_str, index=False)
    test_df.to_csv(test_str, index=False)

    return {
        "test": test_str.getvalue(),
        "train": train_str.getvalue()
    }



@app.route('/train', methods=['POST'])
def train():
    record = json.loads(request.data)
    df = pd.read_csv(io.StringIO(record["dataset"]))
    target_name = record["target_name"]
    ignored_columns = record["ignored_columns"]
    model = train_model(df, target_name, ignored_columns)
    stream = io.BytesIO()
    pickle.dump(model, stream)
    bytes = stream.getvalue()
    encodedBytes = base64.b64encode(bytes)
    model_str = str(encodedBytes, "utf-8")
    return {
        "model": model_str
    }

if __name__ == '__main__':
    krr = joblib.load("/Users/Chris/PycharmProjects/Beam/q1/model1.pk1")  # Load "model.pkl"
    print('Model loaded')
    app.run(debug=True,port=5000)
