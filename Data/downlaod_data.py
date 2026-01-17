import pandas as pd

train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# print(train_df.head())

# save to csv 
train_df.to_csv("data/banking_train.csv", index=False)
test_df.to_csv("data/banking_test.csv", index=False)