#%%
import pandas as pd

# 검증용 feature 데이터 불러오기
test_feature = pd.read_csv("./data/test_feature.csv")

test_feature["일시"] = pd.to_datetime(test_feature["일시"])
test_feature["월"] = 0
test_feature["년도"] = 0
test_feature["월"] = test_feature["일시"].dt.month
test_feature["년도"] = test_feature["일시"].dt.year


test_feature = test_feature.rename(
    columns={
        "평균기온(°C)": "평균기온",
        "평균최고기온(°C)": "평균최고기온",
        "평균최저기온(°C)": "평균최저기온",
        "평균상대습도(%)": "평균상대습도",
        "평균운량(1/10)": "평균운량",
        "합계 일조시간(hr)": "합계 일조시간",
    },
)

# 구름 분류
tiny_cloud_cond = 3.9  # 25% 값
many_cloud_cond = 5.3  # 75% 값

test_feature["cloud_amount"] = 0

tiny_cloud_cond_index = test_feature[
    test_feature["평균운량"] <= tiny_cloud_cond
].index

normal_cloud_cond_index = test_feature[
    (test_feature["평균운량"] < many_cloud_cond)
    & (test_feature["평균운량"] > tiny_cloud_cond)
].index

many_cloud_cond_index = test_feature[
    test_feature["평균운량"] >= many_cloud_cond
].index

test_feature["cloud_amount"].iloc[tiny_cloud_cond_index] = 1
test_feature["cloud_amount"].iloc[normal_cloud_cond_index] = 2
test_feature["cloud_amount"].iloc[many_cloud_cond_index] = 3
del test_feature["평균운량"]

# 일조량 분류
tiny_sun_cond = 183  # 25% 값
many_sun_cond = 236  # 75% 값

test_feature["sun_amount"] = 0

tiny_sun_cond_index = test_feature[
    test_feature["합계 일조시간"] <= tiny_sun_cond
].index

normal_sun_cond_index = test_feature[
    (test_feature["합계 일조시간"] < many_sun_cond)
    & (test_feature["합계 일조시간"] > tiny_sun_cond)
].index

many_sun_cond_index = test_feature[
    test_feature["합계 일조시간"] >= many_sun_cond
].index

test_feature["sun_amount"].iloc[tiny_sun_cond_index] = 1
test_feature["sun_amount"].iloc[normal_sun_cond_index] = 2
test_feature["sun_amount"].iloc[many_sun_cond_index] = 3
del test_feature["합계 일조시간"]


test_feature = pd.get_dummies(test_feature, columns=["월"])

# 불쾌지수
test_feature["불쾌지수"] = (
    0.81 * test_feature["평균기온"]
    + 0.01 * test_feature["평균상대습도"] * (0.99 * test_feature["평균기온"] - 14.3)
    + 46.3
)

del test_feature["지점"]
del test_feature["지점명"]
del test_feature["일시"]
del test_feature["월합강수량(00~24h만)(mm)"]
del test_feature["평균최저기온"]
del test_feature["평균최고기온"]
del test_feature["년도"]

# 검증용 target 데이터 불러오기
test_target = pd.read_csv("./data/test_target.txt", sep="\t")

drop_index = test_target[test_target["구분"] != "합계"].index
test_target = test_target.drop(drop_index)
test_target["합계"] = test_target["합계"].apply(lambda x: x.replace(",", ""))
test_target["합계"] = test_target["합계"].astype(float)
test_target = test_target.reset_index(drop=True)
test_target = test_target["합계"]
test_feature = test_feature.astype(
    {
        "평균기온": "float64",
        "월_1": "int64",
        "월_2": "int64",
        "월_3": "int64",
        "월_4": "int64",
        "월_5": "int64",
        "월_6": "int64",
        "월_7": "int64",
        "월_8": "int64",
        "월_9": "int64",
        "월_10": "int64",
        "월_11": "int64",
        "월_12": "int64",
    }
)
test_feature.info()
#%%
import joblib

# 모델 불러오기
ridge = joblib.load("./ridge_model.pkl")
randomforest_model = joblib.load("./randomforest_model.pkl")
elasic_model = joblib.load("./elasic_model.pkl")


# 검증용 데이터 정규화
import joblib
import numpy as np

file_name = "StandardScaler.pkl"
sc = joblib.load(file_name)
scaled_test_feature = sc.transform(test_feature)
scaled_test_target = np.log1p(test_target)

#%%
# ridge 검증용 데이터 예측
pred = ridge.predict(scaled_test_feature)
print(np.round(np.expm1(pred), 0))

# 검증용 데이터 예측 점수
score = ridge.score(scaled_test_feature, scaled_test_target)
print(score)
#%%
# 랜덤 포레스트
pred = randomforest_model.predict(scaled_test_feature)
print(np.round(np.expm1(pred), 0))

# 검증용 데이터 예측 점수
score = randomforest_model.score(scaled_test_feature, scaled_test_target)
print(score)
#%%
# 엘라스틱넷
pred = elasic_model.predict(scaled_test_feature)
print(np.round(np.expm1(pred), 0))

# 검증용 데이터 예측 점수
score = elasic_model.score(scaled_test_feature, scaled_test_target)
print(score)
# %%
# 지점,지점명,일시,평균기온(°C),평균최고기온(°C),평균최저기온(°C),평균상대습도(%),평균운량(1/10),합계 일조시간(hr)
# 108,서울,2014-09,22.1,27,18,69,4.9,214.3
import joblib
import pandas as pd

ridge = joblib.load("./ridge_model.pkl")

X_feature = pd.DataFrame(
    data={
        "평균기온": [-0.7],
        "평균상대습도": [50],
        "cloud_amount": [1],
        "sun_amount": [2],
        "월_1": [1],
        "월_2": [0],
        "월_3": [0],
        "월_4": [0],
        "월_5": [0],
        "월_6": [0],
        "월_7": [0],
        "월_8": [0],
        "월_9": [0],
        "월_10": [0],
        "월_11": [0],
        "월_12": [0],
    }
)

X_feature = X_feature.astype(
    {
        "평균기온": "float64",
        "월_1": "int64",
        "월_2": "int64",
        "월_3": "int64",
        "월_4": "int64",
        "월_5": "int64",
        "월_6": "int64",
        "월_7": "int64",
        "월_8": "int64",
        "월_9": "int64",
        "월_10": "int64",
        "월_11": "int64",
        "월_12": "int64",
    }
)
X_feature["불쾌지수"] = (
    0.81 * X_feature["평균기온"]
    + 0.01 * X_feature["평균상대습도"] * (0.99 * X_feature["평균기온"] - 14.3)
    + 46.3
)
X_feature.info()
# %%
import joblib

scaled_X_feature = sc.transform(X_feature)

# 검증용 데이터 예측
#%%
pred = ridge.predict(scaled_X_feature)
print(np.round(np.expm1(pred), 0))


# %%
