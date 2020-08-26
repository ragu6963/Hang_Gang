#%%
import pandas as pd

#%%
weahter_df = pd.read_csv(f"./data/weather.csv")
weahter_df
#%%
weahter_df["일시"] = pd.to_datetime(weahter_df["일시"])
weahter_df["월"] = 0
weahter_df["년도"] = 0
weahter_df["월"] = weahter_df["일시"].dt.month
weahter_df["년도"] = weahter_df["일시"].dt.year
del weahter_df["지점"]
del weahter_df["지점명"]
del weahter_df["일시"]
weahter_df = weahter_df.rename(
    columns={
        "평균기온(°C)": "평균기온",
        "평균최고기온(°C)": "평균최고기온",
        "평균최저기온(°C)": "평균최저기온",
        "평균상대습도(%)": "평균상대습도",
        "평균운량(1/10)": "평균운량",
        "합계 일조시간(hr)": "합계 일조시간",
    },
)

#%%
weahter_df.head(12)
# %%
weahter_df.to_csv(f"feature.csv", index=False)

# %%
