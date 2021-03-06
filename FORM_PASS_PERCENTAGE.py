import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Book3.csv")
label_encoder = LabelEncoder()

dataset = pd.read_csv("Book3.csv")
dataset = dataset[dataset["ODK"] == "D"]

df = data[["PLAY TYPE","OFF FORM"]]
df = df[df["PLAY TYPE"] == "Pass"]
df["PLAY TYPE PASS"] = 1

pass_num = df.groupby(["OFF FORM","PLAY TYPE PASS"])

d1 = pass_num['PLAY TYPE PASS'].sum().to_frame(name = 'COUNT').reset_index()
print(d1)

df = data[["PLAY TYPE","OFF FORM"]]
df = df[df["PLAY TYPE"] == "Run"]
df["PLAY TYPE RUN"] = 1

run_num = df.groupby(["OFF FORM","PLAY TYPE RUN"])

d2 = run_num['PLAY TYPE RUN'].sum().to_frame(name = 'COUNT').reset_index()
print(d2)

d3=d2

d3 = d3.drop(columns="PLAY TYPE RUN")
print(d3)

for i in range(len(d3)):
    if (d1.iloc[i, 0] == d3[i, 0]):
        d3["pass count"] = d1.iloc[i, 2]

print(d3)