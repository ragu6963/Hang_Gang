#%%
import pandas as pd

# %%
daegu_df = pd.read_csv("./data/daegu.csv",)

del daegu_df["지점"]
del daegu_df["지점명"]
daegu_df = daegu_df.rename(
    columns={"평균기온(°C)": "대구기온", "일강수량(mm)": "대구강수량", "평균 풍속(m/s)": "대구풍속",}
)
#%%
daegu_day_df = daegu_df.groupby("일시", as_index=False).mean()
daegu_day_df["주"] = 0
#%%
daegu_day_df
# %%
week = 1
count = 1
for i in range(len(daegu_day_df)):
    daegu_day_df["주"].iloc[i] = week
    count += 1
    if count == 8:
        count = 1
        week += 1

# %%
daegu_day_df["월"] = 0
#%%
daegu_day_df["일시"] = pd.to_datetime(daegu_day_df["일시"])
#%%
daegu_day_df["월"] = daegu_day_df["일시"].dt.month
# %%
daegu_week_df = daegu_day_df.groupby(["주", "월"], as_index=False).mean()
#%%
daegu_week_df

# ----------------------------------------------------------------
# %%
daegeon_df = pd.read_csv("./data/daegeon.csv",)

del daegeon_df["지점"]
del daegeon_df["지점명"]
daegeon_df = daegeon_df.rename(
    columns={"평균기온(°C)": "대전기온", "일강수량(mm)": "대전강수량", "평균 풍속(m/s)": "대전풍속",}
)
#%%
daegeon_day_df = daegeon_df.groupby("일시", as_index=False).mean()
daegeon_day_df["주"] = 0
#%%
daegeon_day_df
# %%
week = 1
count = 1
for i in range(len(daegeon_day_df)):
    daegeon_day_df["주"].iloc[i] = week
    count += 1
    if count == 8:
        count = 1
        week += 1

# %%
daegeon_day_df["월"] = 0
#%%
daegeon_day_df["일시"] = pd.to_datetime(daegeon_day_df["일시"])
#%%
daegeon_day_df["월"] = daegeon_day_df["일시"].dt.month
# %%
daegeon_week_df = daegeon_day_df.groupby(["주", "월"], as_index=False).mean()
#%%
daegeon_week_df

# %%
join_df = pd.merge(daegu_week_df, daegeon_week_df, on=["주", "월"], how="left")


# %%
join_df.head(20)
# %%
