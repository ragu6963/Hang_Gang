#%%
import pandas as pd

feature_df = pd.read_csv(f"./data/train_feature.csv")
#%%
weather_df = feature_df.copy()
weather_df["일시"] = pd.to_datetime(weather_df["일시"])
weather_df["월"] = 0
weather_df["년도"] = 0
weather_df["월"] = weather_df["일시"].dt.month
weather_df["년도"] = weather_df["일시"].dt.year


weather_df = weather_df.rename(
    columns={
        "평균기온(°C)": "평균기온",
        "평균최고기온(°C)": "평균최고기온",
        "평균최저기온(°C)": "평균최저기온",
        "평균상대습도(%)": "평균상대습도",
        "평균운량(1/10)": "평균운량",
        "합계 일조시간(hr)": "합계 일조시간",
        "월합강수량(00~24h만)(mm)": "월합강수량",
    },
)

# 구름양 판단 데이터 컬럼
# 구름이 많다 보통이다 적다
tiny_cloud_cond = 3.9  # 25% 값
many_cloud_cond = 5.3  # 75% 값

weather_df["cloud_amount"] = 0

tiny_cloud_cond_index = weather_df[weather_df["평균운량"] <= tiny_cloud_cond].index

normal_cloud_cond_index = weather_df[
    (weather_df["평균운량"] < many_cloud_cond)
    & (weather_df["평균운량"] > tiny_cloud_cond)
].index

many_cloud_cond_index = weather_df[weather_df["평균운량"] >= many_cloud_cond].index

weather_df["cloud_amount"].iloc[tiny_cloud_cond_index] = 1
weather_df["cloud_amount"].iloc[normal_cloud_cond_index] = 2
weather_df["cloud_amount"].iloc[many_cloud_cond_index] = 3
del weather_df["평균운량"]

# 일조량 판단 데이터 컬럼
# 일조량이 많다 보통이다 적다
tiny_sun_cond = 183  # 25% 값
many_sun_cond = 236  # 75% 값
weather_df["sun_amount"] = 0
tiny_sun_cond_index = weather_df[weather_df["합계 일조시간"] <= tiny_sun_cond].index

normal_sun_cond_index = weather_df[
    (weather_df["합계 일조시간"] < many_sun_cond)
    & (weather_df["합계 일조시간"] > tiny_sun_cond)
].index

many_sun_cond_index = weather_df[weather_df["합계 일조시간"] >= many_sun_cond].index

weather_df["sun_amount"].iloc[tiny_sun_cond_index] = 1
weather_df["sun_amount"].iloc[normal_sun_cond_index] = 2
weather_df["sun_amount"].iloc[many_sun_cond_index] = 3
del weather_df["합계 일조시간"]

del weather_df["지점"]
del weather_df["지점명"]
del weather_df["일시"]
del weather_df["평균최저기온"]
del weather_df["평균최고기온"]
del weather_df["월합강수량"]


#%%
weather_df.head(12)
# %%
weather_df.to_csv(f"./data/feature.csv", index=False)

# %%
