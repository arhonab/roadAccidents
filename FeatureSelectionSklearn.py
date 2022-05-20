# from sklearn.datasets import load_iris
# from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np

# feature_names = np.array(diabetes.feature_names)
# print(type(feature_names))
# X, y = load_iris(return_X_y=True)
# diabetes = load_diabetes()
# X, y = diabetes.data, diabetes.target
# print(diabetes.feature_names)


# fatality : PSEV=Outcome 1=0 2=0 3=1
data_org = pd.read_csv(r'Documents\NCDB_2017.csv', dtype="string")

# features = ['P_SEX', 'C_RCFG', 'C_RSUR', 'V_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'C_CONF', 'C_WTHR'];
MAXCOL = 22
# feature_names = data_org.iloc[1,0:MAXCOL]
features = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS', 'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN',
            'C_TRAF', 'V_ID', 'V_TYPE', 'V_YEAR', 'P_ID', 'P_SEX', 'P_AGE', 'P_PSN', 'P_ISEV', 'P_SAFE', 'P_USER'];
for col in features:
    data_org[col] = pd.to_numeric(data_org[col], errors='coerce')
    data_org = data_org[data_org[col] < 77770]

data_filter_count = len(data_org)

print("data_read_count=", data_read_count, "data_filter_count=", data_filter_count)

data_org['Outcome'] = pd.to_numeric(data_org['Outcome'], errors='coerce')
# CHECK
# data_org = data_org[data_org['Outcome'] < 77770]

# ROWS = 0
# ROWE = 1000
# ROWS = 6600
# ROWE = 8100
# print("SBRESULT : data_org read rows={} filtered rows={} processed rows={}".format(data_read_count, len(data_org), ROWE-ROWS))
# data_org = data_org[ROWS:ROWE]

X = data_org.iloc[:, 1:23]
y = data_org.iloc[:, 23]
# print(X.shape)

# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# X_new1 = model.transform(X)
# print(X_new1.shape)


lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

print("Input features:", features)

# print(lasso.coef_)
# print(lasso.intercept_)
# print(lasso)
# importance = np.abs(lasso.coef_)
importance = lasso.coef_
print("importance:", importance)

LAMBDA = 10

# print(type(importance), importance.shape)

VALMIN = -9999

# threshold = np.sort(importance)[-3] + 0.01
# lambda = 25
threshold = np.sort(importance)[-LAMBDA]
# threshold = 8.47462254e-04
# threshold = 4.150183973242545e-07
# threshold = -8.47462254e-04
# threshold = -1
threshold = 0.0
print("Feature selection threshold =", threshold)

print("Selected top 10 features:")

for i in range(LAMBDA):
    max = VALMIN
    maxj = 0
    for j in range(MAXCOL):
        if importance[j] > max:
            max = importance[j]
            maxj = j
    # print("max=", max, "maxj=", maxj, importance[maxj], features[maxj])
    print(features[maxj])
    importance[maxj] = VALMIN

# print("first=", feature_names[0])

# print(feature_names.shape, "feature_names:", feature_names)

# sfm = SelectFromModel(lasso, threshold=threshold).fit(X, y)
# print("Features selected by SelectFromModel: ", sfm.get_support())

# print("SBRESULT: len=", len(features))
# print("SBRESULT: len=", len(feature_names))
# print(feature_names)
# print("Features selected by SelectFromModel: ", feature_names[sfm.get_support()])
# print("Features selected by SelectFromModel: "
#      f"{feature_names[sfm.get_support()]}")

