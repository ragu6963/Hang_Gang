#%%
import pandas as pd

target_df = pd.read_csv("./data/train_target.txt", sep="\t")

# %%
target_df["기간"] = target_df["기간"].apply(lambda x: str(x).ljust(7, "0"))
target_df["기간"] = pd.to_datetime(target_df["기간"], format="%Y.%m")
target_df["월"] = 0
target_df["년도"] = 0
target_df["월"] = target_df["기간"].dt.month
target_df["년도"] = target_df["기간"].dt.year

drop_index = target_df[target_df["구분"] != "합계"].index
target_df = target_df.drop(drop_index)
target_df = target_df.reset_index(drop=True)
del target_df["기간"]
del target_df["구분"]

# target 타입 숫자형으로 변경
target_df["합계"] = target_df["합계"].apply(lambda x: x.replace(",", ""))
target_df["합계"] = target_df["합계"].astype(float)
target_df["이용객 수"] = target_df["합계"]
del target_df["합계"]
del target_df["주요행사 및 마라톤"]
del target_df["일반이용자"]
del target_df["운동시설"]
del target_df["자전거"]
del target_df["특화공원"]
del target_df["기타"]

# %%
target_df.to_csv(f"target.csv", index=False)

# %%
