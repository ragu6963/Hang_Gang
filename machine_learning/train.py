# target == 합계
#%%
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
)
import numpy as np
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
            model = ElasticNet(alpha=param, l1_ratio=0.7, random_state=0)
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

target_feature_df = pd.read_csv("./data/target_feature.csv")

# train_target 데이터 copy
train_target_feature_df = target_feature_df.copy()
#%%
# 불필요한 컬럼 삭제
train_target_feature_df = train_target_feature_df.drop(["년도"], axis=1)

# 이상치 제거
target_cond = train_target_feature_df[
    (train_target_feature_df["이용객 수"] >= 9000000)
    | (train_target_feature_df["이용객 수"] <= 2000000)
].index
train_target_feature_df = train_target_feature_df.drop(target_cond)

# 인덱스 리셋
train_target_feature_df.reset_index(drop=True, inplace=True)

# feature,target 추출
feature = train_target_feature_df.drop("이용객 수", axis=1)
target = train_target_feature_df["이용객 수"]

import joblib

# feature, target 정규화
sc = StandardScaler()
sc.fit(feature)
scaled_feature = sc.transform(feature)
file_name = "StandardScaler.pkl"
joblib.dump(sc, file_name)

scaled_target = np.log1p(target)
#%%
# 릿지 모델 파일생성
import joblib
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    scaled_feature, scaled_target, test_size=0.2, random_state=0
)
ridge = Ridge(alpha=0.07)
ridge.fit(X_train, y_train)
score = ridge.score(X_test, y_test)
print(score, end=" ")
# joblib.dump(ridge, "./ridge_model.pkl")

# %%
alphas = [0.07, 0.1, 0.5, 1, 3, 5, 7]
coeff_elastic_df = get_linear_reg_eval(
    "Ridge",
    alphas,
    X_data_n=scaled_feature,
    y_target_n=scaled_target,
    feature=feature,
    target=target,
)


#%%
# ElasicNet 모델 파일생성
import joblib
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    scaled_feature, scaled_target, test_size=0.2, random_state=0
)
ElasicNet = ElasticNet(alpha=0.07, l1_ratio=0.7, random_state=0)
ElasicNet.fit(X_train, y_train)
score = ElasicNet.score(X_test, y_test)
# joblib.dump(ridge, "./elasic_model.pkl")
print(score)
# %%

alphas = [0.07, 0.1, 0.5, 1, 3, 5, 7]
coeff_elastic_df = get_linear_reg_eval(
    "ElasicNet",
    alphas,
    X_data_n=scaled_feature,
    y_target_n=scaled_target,
    feature=feature,
    target=target,
)

#%%
# 랜덤포레스트 모델 파일생성
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    scaled_feature, scaled_target, test_size=0.2, random_state=0
)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
rf_reg.fit(X_train, y_train)
score = rf_reg.score(X_test, y_test)
joblib.dump(rf_reg, "./randomforest_model.pkl")
print(score)


#%%
# 폰트 설정
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14, 4)

# %%
# 이상치 제거 그래프

fig, axs = plt.subplots(figsize=(18, 12), ncols=4, nrows=1)
features = ["평균기온", "평균상대습도", "불쾌지수", "년도"]
for i, feature in enumerate(features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="이용객 수", data=train_target_feature_df, ax=axs[i])
# %%
# 이상치 미제거 그래프
fig, axs = plt.subplots(figsize=(18, 12), ncols=4, nrows=1)
features = ["평균기온", "평균상대습도", "불쾌지수", "년도"]
for i, feature in enumerate(features):
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature, y="이용객 수", data=target_feature_df, ax=axs[i])
#%%
sns.regplot(x="평균기온", y="이용객 수", data=train_target_feature_df)
#%%
sns.regplot(x="평균상대습도", y="이용객 수", data=train_target_feature_df)
#%%
sns.regplot(x="불쾌지수", y="이용객 수", data=train_target_feature_df)
#%%
sns.regplot(x="년도", y="이용객 수", data=train_target_feature_df)
#%%
sns.scatterplot(x="평균운량", y="이용객 수", data=target_feature_df)
#%%
sns.scatterplot(x="월", y="평균운량", data=target_feature_df)
sns.relplot(x="월", y="이용객 수", data=target_feature_df)

#%%
corr = feature.corr()
plt.figure(figsize=(18, 18))
sns.heatmap(corr, annot=True, fmt=".1g")

# %%
