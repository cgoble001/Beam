import  requests



if __name__ == "__main__":
    with open("./sample_data_format.csv") as f:
        dataset = f.read()
    split_query = {
        "data": dataset,
        "size": 5000
    }
    split_res = requests.post("http://127.0.0.1:5000/split", json=split_query)

    train_query = {
        "target_name": "R_T1W",
        "ignored_columns": ["Date", "Coin"],
        "dataset" : split_res.json()["train"]
    }

    train_res = requests.post("http://127.0.0.1:5000/train", json=train_query)

    test_query = {
        "model": train_res.json()["model"],
        "dataset": split_res.json()["test"]
    }
    test_res = requests.post("http://127.0.0.1:5000/test", json=test_query)

    print(test_res.text)
