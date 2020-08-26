#%%
from os import replace
import pandas as pd

feature_df = pd.read_csv("./feature.csv")
target_df = pd.read_csv("./target.csv")
#%%
feature_df
# %%
target_feature_join_df = pd.merge(feature_df, target_df, on=["월", "년도"])
target_feature_join_df = target_feature_join_df[
    [
        "년도",
        "월",
        "평균기온",
        "평균최고기온",
        "평균최저기온",
        "평균상대습도",
        "평균운량",
        "합계 일조시간",
        "일반이용자",
        "운동시설",
        "자전거",
    ]
]
#%%
target_feature_join_df
# %%
target_feature_join_df.to_csv(f"target_feature.csv", index=False)

# %%
