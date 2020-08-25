#%%
import pandas as pd
import numpy as np

target_list = [
    "cabbage",
    "lettuce",
    "sesame",
]
target_df = pd.read_excel(f"data/{target_list[0]}.xls")

#%%
target_df.rename(columns={"Unnamed: 1": "순", "\t구분": "년도"}, inplace=True)
drop_index = target_df[target_df["순"] == "평균"].index
target_df = target_df.drop(drop_index)
del target_df["연평균"]
del target_df["년도"]
#%%
target_df
# %%
first = target_df.iloc[:3, :]
second = target_df.iloc[3:6, :]
third = target_df.iloc[6:, :]
# %%
join_df = pd.DataFrame()
#%%
first
# %%
count = 1
for i in range(1, 13):
    print(f"{i}월")
    tmp = pd.DataFrame()
    for j in range(0, 3):
        print(first.iloc[j][i], end=" ")
        tmp2 = pd.DataFrame()
        tmp2["순"] = count
        tmp2["가격"] = first.iloc[j][i]
        tmp = pd.concat(tmp, tmp2)
    print()
# %%
