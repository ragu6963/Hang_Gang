#%%
import pandas as pd

feature_df = pd.read_csv("./data/feature.csv")
target_df = pd.read_csv("./data/target.csv")

target_feature_join_df = pd.merge(feature_df, target_df, on=["월", "년도"])

# 월 데이터 원핫 인코딩
target_feature_join_df = pd.get_dummies(target_feature_join_df, columns=["월"])

# 불쾌지수 계산
target_feature_join_df["불쾌지수"] = (
    0.81 * target_feature_join_df["평균기온"]
    + 0.01
    * target_feature_join_df["평균상대습도"]
    * (0.99 * target_feature_join_df["평균기온"] - 14.3)
    + 46.3
)
target_feature_join_df
# %%
target_feature_join_df.to_csv(f"./data/target_feature.csv", index=False)

# %%
