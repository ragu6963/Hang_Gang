#%%
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
)
import numpy as np


def get_scaled_data(method="None", p_degree=None, input_data=None):
    if method == "Standard":
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == "MinMax":
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == "Log":
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(
            degree=p_degree, include_bias=False
        ).fit_transform(scaled_data)

    return scaled_data


from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import cross_val_score


def get_linear_reg_eval(
    model_name,
    params=None,
    X_data_n=None,
    y_target_n=None,
    verbose=True,
    feature=None,
    target=None,
):
    coeff_df = pd.DataFrame()
    if verbose:
        print(f"##### {model_name} #####")
    for param in params:
        if model_name == "Ridge":
            model = Ridge(alpha=param, random_state=0)
        elif model_name == "Lasso":
            model = Lasso(alpha=param, random_state=0)
        elif model_name == "ElasicNet":
            model = ElasticNet(alpha=param, l1_ratio=0.6, random_state=0)
        neg_mse_scores = cross_val_score(
            model, X_data_n, y_target_n, scoring="neg_mean_squared_error", cv=5
        )
        r2_score = cross_val_score(
            model, X_data_n, y_target_n, scoring="r2", cv=5
        )
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        avg_r2 = np.mean(r2_score)
        print(
            f"alpha : {param}, cv : 5, 평균 RMSE : {avg_rmse:.2f}, Variance socre : {avg_r2:.2f}"
        )
        model.fit(feature, target)
        coeff = pd.Series(data=model.coef_, index=feature.columns)
        colname = "alpha:" + str(param)
        coeff_df[colname] = coeff
    return coeff_df


#%%
# 데이터 불러오기
import pandas as pd

target_feature_df = pd.read_csv("./target_feature.csv")

# target 타입 숫자형으로 변경
target_feature_df["일반이용자"] = target_feature_df["일반이용자"].apply(
    lambda x: x.replace(",", "")
)
target_feature_df["운동시설"] = target_feature_df["운동시설"].apply(
    lambda x: x.replace(",", "")
)
target_feature_df["자전거"] = target_feature_df["자전거"].apply(
    lambda x: x.replace(",", "")
)

target_feature_df["자전거"] = target_feature_df["자전거"].astype(float)
target_feature_df["일반이용자"] = target_feature_df["일반이용자"].astype(float)
target_feature_df["운동시설"] = target_feature_df["운동시설"].astype(float)

# 불필요한 컬럼 삭제
del target_feature_df["평균최고기온"]
del target_feature_df["평균최저기온"]


#%%
# 구름양 판단 데이터 컬럼
# 구름이 많다 보통이다 적다
target_feature_df["many_cloud"] = 0
target_feature_df["normal_cloud"] = 0
target_feature_df["tiny_cloud"] = 0

tiny_cloud_cond = 3.9  # 25% 값
many_cloud_cond = 5.3  # 75% 값
tiny_cloud_cond_index = target_feature_df[
    target_feature_df["평균운량"] <= tiny_cloud_cond
].index

normal_cloud_cond_index = target_feature_df[
    (target_feature_df["평균운량"] < many_cloud_cond)
    & (target_feature_df["평균운량"] > tiny_cloud_cond)
].index

many_cloud_cond_index = target_feature_df[
    target_feature_df["평균운량"] >= many_cloud_cond
].index

target_feature_df["tiny_cloud"].iloc[tiny_cloud_cond_index] = 1
target_feature_df["normal_cloud"].iloc[normal_cloud_cond_index] = 1
target_feature_df["many_cloud"].iloc[many_cloud_cond_index] = 1
del target_feature_df["평균운량"]

#%%
# 일조량 판단 데이터 컬럼
# 일조량이 많다 보통이다 적다
target_feature_df["many_sun"] = 0
target_feature_df["normal_sun"] = 0
target_feature_df["tiny_sun"] = 0

tiny_sun_cond = 183  # 25% 값
many_sun_cond = 236  # 75% 값
tiny_sun_cond_index = target_feature_df[
    target_feature_df["합계 일조시간"] <= tiny_sun_cond
].index

normal_sun_cond_index = target_feature_df[
    (target_feature_df["합계 일조시간"] < many_sun_cond)
    & (target_feature_df["합계 일조시간"] > tiny_sun_cond)
].index

many_sun_cond_index = target_feature_df[
    target_feature_df["합계 일조시간"] >= many_sun_cond
].index

target_feature_df["tiny_sun"].iloc[tiny_sun_cond_index] = 1
target_feature_df["normal_sun"].iloc[normal_sun_cond_index] = 1
target_feature_df["many_sun"].iloc[many_sun_cond_index] = 1
del target_feature_df["합계 일조시간"]

#%%
target_feature_df.describe()
#%%
# feature target 저장용 list
target_name_list = ["일반이용자", "자전거", "운동시설"]
target_list = []
feature_list = []

#%%
# 일반이용자 학습용 데이터프레임 가공
target1_feature_df = target_feature_df.drop(
    ["일반이용자", "운동시설", "자전거", "년도"], axis=1
)

# 월 데이터 원핫 인코딩
target1_feature_df = pd.get_dummies(target1_feature_df, columns=["월"])

target1_feature_df["target"] = target_feature_df["일반이용자"]
# 일반이용자 이상치 제거
target_1_cond = target1_feature_df[
    (target1_feature_df["target"] >= 3500000)
    | (target1_feature_df["target"] <= 1000000)
].index
target1_feature_df = target1_feature_df.drop(target_1_cond)


# feature,target 추출
feature = target1_feature_df.drop("target", axis=1)
target = target1_feature_df["target"]

# list에 저장
target_list.append(target)
feature_list.append(feature)

#%%
# 자전거 학습용 데이터프레임 가공
target2_feature_df = target_feature_df.drop(
    ["일반이용자", "운동시설", "자전거", "년도"], axis=1
)


# 월 데이터 원핫 인코딩
target2_feature_df = pd.get_dummies(target2_feature_df, columns=["월"])

target2_feature_df["target"] = target_feature_df["자전거"]

# 자전거 이상치 제거
target_2_cond = target2_feature_df[
    (target2_feature_df["target"] >= 2500000)
    | (target2_feature_df["target"] <= 500000)
].index
target2_feature_df = target2_feature_df.drop(target_2_cond)

# feature,target 추출
feature = target2_feature_df.drop("target", axis=1)
target = target2_feature_df["target"]

# list에 저장
target_list.append(target)
feature_list.append(feature)
#%%
# 운동시설 학습용 데이터프레임
target3_feature_df = target_feature_df.drop(
    ["일반이용자", "운동시설", "자전거", "년도"], axis=1
)

# 월 데이터 원핫 인코딩
target3_feature_df = pd.get_dummies(target3_feature_df, columns=["월"])

target3_feature_df["target"] = target_feature_df["운동시설"]


# 운동시설 이상치 제거
target_3_cond = target3_feature_df[
    (target3_feature_df["target"] >= 1000000)
    | (target3_feature_df["target"] <= 200000)
].index
target3_feature_df = target3_feature_df.drop(target_3_cond)

# feature,target 추출
feature = target3_feature_df.drop("target", axis=1)
target = target3_feature_df["target"]

# list에 저장
target_list.append(target)
feature_list.append(feature)
#%%
# feature, target 정규화
scaled_feature_list = []
scaled_target_list = []
for target in target_list:
    scaled_target = get_scaled_data("Log", None, target)
    scaled_target_list.append(scaled_target)

for feature in feature_list:
    scaled_feature = get_scaled_data("Standard", None, feature)
    scaled_feature_list.append(scaled_feature)


#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


for index in range(3):
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_feature_list[index],
        scaled_target_list[index],
        test_size=0.2,
        random_state=156,
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(f"*****{target_name_list[index]}*****")
    print("MSE : {0:.3f} , RMSE : {1:.3F}".format(mse, rmse))
    print("Variance score : {0:.3f}".format(r2_score(y_test, pred)))

# %%
for index in range(3):
    alphas = [0.07, 0.1, 0.5, 1, 3, 5, 7]
    print(f"*****{target_name_list[index]}*****")
    coeff_lasso_df = get_linear_reg_eval(
        "Lasso",
        alphas,
        X_data_n=scaled_feature_list[index],
        y_target_n=scaled_target_list[index],
        feature=feature_list[index],
        target=target_list[index],
    )
    print()

# %%
for index in range(3):
    print(f"*****{target_name_list[index]}*****")
    alphas = [0.07, 0.1, 0.5, 1, 3, 5, 7]
    coeff_elastic_df = get_linear_reg_eval(
        "ElasicNet",
        alphas,
        X_data_n=scaled_feature_list[index],
        y_target_n=scaled_target_list[index],
        feature=feature_list[index],
        target=target_list[index],
    )
    print()

# %%
for index in range(3):
    print(f"*****{target_name_list[index]}*****")
    alphas = [0.07, 0.1, 0.5, 1, 3, 5, 7]
    coeff_elastic_df = get_linear_reg_eval(
        "Ridge",
        alphas,
        X_data_n=scaled_feature_list[index],
        y_target_n=scaled_target_list[index],
        feature=feature_list[index],
        target=target_list[index],
    )
    print()

#%%
from sklearn.decomposition import PCA

pca_value_list = [4, 6, 8, 10, 12, 14, 16, 18]
for pca_value in pca_value_list:
    pca = PCA(n_components=pca_value)
    # fit 과 transform을 호출해 PCA 변환
    pca.fit(scaled_feature_list[0])
    feature_pca = pca.transform(scaled_feature_list[0])
    # print(feature_pca.shape)
    # print(pca.explained_variance_ratio_)

    model = Ridge(alpha=0.07, random_state=0)
    neg_mse_scores = cross_val_score(
        model,
        feature_pca,
        scaled_target_list[0],
        scoring="neg_mean_squared_error",
        cv=5,
    )

    neg_mse_scores = cross_val_score(
        model,
        feature_pca,
        scaled_target_list[0],
        scoring="neg_mean_squared_error",
        cv=5,
    )
    r2_score = cross_val_score(
        model, feature_pca, scaled_target_list[0], scoring="r2", cv=5
    )
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    avg_r2 = np.mean(r2_score)
    print(
        f"pca : {pca_value}, cv : 5, 평균 RMSE : {avg_rmse:.2f}, Variance socre : {avg_r2:.2f}"
    )
################################################################
# 이후 그래프
#%%
# 폰트 설정
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14, 4)

# %%
# 일반이용자 이상치 제거 그래프
fig, axs = plt.subplots(figsize=(18, 12), ncols=2, nrows=1)
lm_features = ["평균기온", "평균상대습도"]
for i, feature in enumerate(lm_features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="target", data=target1_feature_df, ax=axs[i])
# %%
# 일반이용자 이상치 미제거 그래프
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(18, 12), ncols=5, nrows=1)
lm_features = ["평균기온", "평균운량", "평균상대습도", "합계 일조시간", "월"]
for i, feature in enumerate(lm_features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="일반이용자", data=target_feature_df, ax=axs[i])

#%%
# 자전거 이상치 제거 그래프
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(18, 12), ncols=2, nrows=1)
lm_features = ["평균기온", "평균상대습도"]
for i, feature in enumerate(lm_features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="target", data=target2_feature_df, ax=axs[i])
# %%
# 일반이용자 자전거 미제거 그래프
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(18, 12), ncols=5, nrows=1)
lm_features = ["평균기온", "평균운량", "평균상대습도", "합계 일조시간", "월"]
for i, feature in enumerate(lm_features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="자전거", data=target_feature_df, ax=axs[i])

# %%
# 운동시설 이상치 제거 그래프
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(18, 12), ncols=2, nrows=1)
lm_features = [
    "평균기온",
    "평균상대습도",
]
for i, feature in enumerate(lm_features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="target", data=target3_feature_df, ax=axs[i])
# %%
# 운동시설 자전거 미제거 그래프
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize=(18, 12), ncols=5, nrows=1)
lm_features = ["평균기온", "평균운량", "평균상대습도", "합계 일조시간", "월"]
for i, feature in enumerate(lm_features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="운동시설", data=target_feature_df, ax=axs[i])

# %%
