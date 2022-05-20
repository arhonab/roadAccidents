import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=plt.cbook.mplDeprecation)
plt.rcParams['figure.max_open_warning'] = 0

# input : "D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.csv"
print("LogisticRegression")

# Load data from the file
# Load the data from file
# data = pd.read_csv("scores_Logistic_Regression.csv")
# data = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017-7-outcome.csv', dtype="string")
# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017-7-100-outcome.csv', dtype="string")
# data = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017-7-500-outcome.csv', dtype="string")
# data = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017-7-1000-outcome.csv', dtype="string")
# data = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017-7-10000-outcome.csv', dtype="string")
# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017-23-100-outcome.csv', dtype="string")

# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.23-numeric-outcome1-fatal.csv', dtype="string")
# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.23-10000-numeric-outcome1-fatal.csv', dtype="string")
# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.23-1000-numeric-outcome1-fatal.csv', dtype="string")

# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.23-numeric-outcome1-injured.csv', dtype="string")
# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.23-1000-numeric-outcome1-injured.csv', dtype="string")
# data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.23-10000-numeric-outcome1-injured.csv', dtype="string")
data_org = pd.read_csv(r'D:\LOCAL\AcadAB\WWSEF\NCDB\NCDB_2017.23-10000-numeric-outcome3-injured.csv', dtype="string")

# print("\n\nOriginal Dataset: ")
# print(data_org.head())

# features = ['C_WTHR', 'V_YEAR', 'P_AGE']
# features = ['V_YEAR', 'P_AGE']
# features = ['V_YEAR', 'P_AGE', 'V_YEAR']
# features = ['C_RCFG', 'C_WTHR', 'C_RSUR', 'V_YEAR', 'P_AGE', 'P_ISEV', 'P_SAFE']
# features = ['C_RCFG', 'C_WTHR', 'C_RSUR', features = ['V_YEAR', 'P_AGE', 'V_YEAR']'V_YEAR', 'P_AGE', 'P_ISEV', 'P_SAFE']
# features = ['C_RCFG', 'C_WTHR', 'C_RSUR', 'V_YEAR', 'P_AGE', 'P_SAFE']
# C_YEAR C_MNTH C_WDAY C_HOUR C_SEV C_VEHS C_CONF C_RCFG C_WTHR C_RSUR C_RALN C_TRAF V_ID V_TYPE V_YEAR P_ID P_SEX P_AGE P_PSN P_ISEV P_SAFE P_USER C_CASE Outcome
# features = ['C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF', 'V_TYPE', 'V_YEAR', 'P_SEX', 'P_AGE', 'P_SAFE']

# featuresf = ['P_USER', 'C_RALN', 'P_SAFE', 'C_WDAY', 'P_PSN', 'P_AGE', 'C_CONF', 'C_YEAR', 'C_MNTH', 'C_HOUR'];
# featuresf = ['P_USER', 'C_RALN', 'P_SAFE', 'C_WDAY', 'P_PSN', 'P_AGE', 'C_CONF', 'C_MNTH', 'C_HOUR', 'C_RALN'];
# features = ['P_USER', 'C_RCFG', 'P_SAFE', 'C_WDAY', 'P_AGE', 'P_PSN', 'C_CONF', 'C_HOUR', 'C_WTHR', 'C_RALN'];

# featuresi = ['P_SEX', 'C_RSUR', 'V_YEAR', 'C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'C_CONF', 'C_RCFG'];
# featuresi = ['P_SEX', 'C_RSUR', 'V_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'C_CONF', 'C_RCFG', 'C_WTHR'];

# features = ['P_SEX', 'C_RSUR', 'V_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'C_CONF', 'C_RCFG', 'C_WTHR'];
features = ['P_SEX', 'C_RCFG', 'C_RSUR', 'V_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'C_CONF', 'C_WTHR'];

# Let us separate the explanatory and dependent variables.
# explanatory = ['Homework Score', 'Time Studied (hours)']
# dependent = ['Outcome']
# explanatory = ['C_RCFG', 'C_WTHR', 'C_RSUR', 'V_YEAR', 'P_AGE', 'P_SAFE']
# features = ['C_RCFG', 'C_WTHR', 'C_RSUR', 'V_YEAR', 'P_AGE', 'P_ISEV', 'P_SAFE', 'Outcome']
# dependent = ['P_ISEV']
# dependent = ['C_SEV']


data_read_count = len(data_org)

"""
data_org['P_SEX'].replace('F', '1', inplace=True)
data_org['P_SEX'].replace('M', '2', inplace=True)
for col in features:
    data_org = data_org[data_org[col] != 'N']
    data_org = data_org[data_org[col] != 'Q']
    data_org = data_org[data_org[col] != 'U']
    data_org = data_org[data_org[col] != 'X']
    data_org = data_org[data_org[col] != 'NN']
    data_org = data_org[data_org[col] != 'QQ']
    data_org = data_org[data_org[col] != 'UU']
    data_org = data_org[data_org[col] != 'XX']
    data_org = data_org[data_org[col] != 'NNNN']
    data_org = data_org[data_org[col] != 'QQQQ']
    data_org = data_org[data_org[col] != 'UUUU']
    data_org = data_org[data_org[col] != 'XXXX']
    data_org[col] = pd.to_numeric(data_org[col], errors='coerce')
"""
for col in features:
    data_org[col] = pd.to_numeric(data_org[col], errors='coerce')
    data_org = data_org[data_org[col] < 77770]

# data_org['V_YEAR'] = [2021-x for x in data_org['V_YEAR']]

data_org['Outcome'] = pd.to_numeric(data_org['Outcome'], errors='coerce')
data_org = data_org[data_org['Outcome'] < 77770]

# Use first 10000 records
# NUMREC = 10000
# NUMREC = 1000
# NUMREC = 1000

# ROWS = 0
# ROWE = 1000
# ROWS = 6600
# ROWE = 8100
ROWS = 0
ROWE = 1500
print("SBRESULT : data_org read rows={} filtered rows={} processed rows={}".format(data_read_count, len(data_org),
                                                                                   ROWE - ROWS))
data_org = data_org[ROWS:ROWE]

# Setup dependent
dependent = ['Outcome']
Yorg = data_org[dependent]

# Right now the class label (outcome) is a text column
# We should convert it to a numerical value
# Fatal = 1, Safe = 0
# y = 1.0*(y == 'Fatal')
# y = 1.0*(y == 'Injury')
# y = 1.0*(y == 'Safe')
# y = 1*(y == 'Safe')
# y = 1*(y == '0')

# Let's look at the transformed target column now
# print("\n\nTarget variable transformed:")
# print(y.head())

# Let's transform the shape of y for future use
# Number of observations
N = len(data_org)
Y = Yorg.values.reshape(N)
Y = Y.astype('int')
# print(Y)

# print("\n\nProcessed Dataset: ")
# print(data_org.head())

# col1 = 'V_YEAR'
# col2 = 'P_AGE'
# explanatory = ['V_YEAR', 'P_AGE']
count = len(features)
# for i in range(0, count-1):
for i in range(0, count - 1):
    col1 = features[i]
    for j in range(i + 1, count):
        col2 = features[j]
        if (col1 == col2):
            continue
        print("col1 = ", col1, "col2=", col2)
        explanatory = [col1, col2]
        # explanatory = [col2, col1]
        # dependent = ['Outcome']

        # print("len col1 = ", len(data[col1]))
        # print("len col2 = ", len(data[col2]))

        # %%
        # Reset data_org
        data = data_org

        X = data[explanatory]
        # y = data[dependent]
        # y = 1*(y == '0')

        # print("N=", N, "len = ", len(y))

        # %%

        # Now, let's visualize the data
        # print("DEBUG : before plt.figure")
        plt.figure()

        # """
        # print("DEBUG : before sns.scatterplot")
        sns.scatterplot(X[col1],
                        X[col2],
                        legend="auto",
                        hue=data[dependent].to_numpy().reshape(N).tolist(),
                        s=100)
        plt.xlabel(col1)
        plt.ylabel(col2)
        # """

        # %%

        # Instantiate the Logistic Regression model
        # LR = LogisticRegression(solver='lbfgs')
        LR = LogisticRegression(C=1e5)
        LR.fit(X, Y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        x_min, x_max = X[col1].min() - 1, X[col1].max() + 1
        y_min, y_max = X[col2].min() - 1, X[col2].max() + 1

        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = LR.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        # plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        # plt.scatter(X[col1], X[col2], c=Y, edgecolors='k', facecolors='r', cmap=plt.cm.Paired)
        scatter = plt.scatter(X[col1], X[col2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel(col1)
        plt.ylabel(col2)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        # plt.xticks()
        # plt.xticks(ticks=[1, 2])
        # plt.xticks(list(set(X[col1])))
        plt.xticks(np.arange(x_min, x_max, step=1))
        plt.yticks()
        # plt.yticks(list(set(X[col2])))
        # Legend
        plt.legend(*scatter.legend_elements(), loc="upper right", title="Outcome")
        # Count
        plt.show()

        # We can find the equation of the line separating
        # the two classes using the intercept_ and coef_
        # attributes of the model
        intercept = LR.intercept_[0]
        coef = LR.coef_[0]

        """
        # Let's plot the decision boundary
        dim1_min = X[col1].min()
        dim1_max = X[col1].max()
        dim2_min = X[col2].min()
        dim2_max = X[col2].max()

        # Build the decision boundary
        y_dim1_min = -intercept/coef[1] - coef[0]*dim1_min/coef[1]
        y_dim1_max = -intercept/coef[1] - coef[0]*dim1_max/coef[1]

        # Plot the line
        plt.fill_between([dim1_min,dim1_max],
                         [y_dim1_min,y_dim1_max],
                         alpha = 0.5,
                         color = 'gray')
        plt.xlim([dim1_min, dim1_max + 1])
        plt.ylim([0, dim2_max + 1])
        """

        # %%

        # Let's add our prediction back to the original dataset
        data['Pred'] = LR.predict(X)

        # Let's figure out which one's we predicted correctly
        data['Correct'] = 1.0 * (data['Pred'] == Y)

        # Determine the accuracy of the model
        accuracy = round(100 * data['Correct'].mean())
        print("\nThe model made correct predictions {}% of the time.".format(accuracy))

        ## Display RESULTS
        # print("SBRESULT : N={} Pred={} Correct={}".format(N, len(data['Pred']), len(data['Correct'])))
        # print(Yorg.head(20))
        ##print("SBRESULT : Yorg type = ", type(Yorg), Yorg.columns, Yorg.dtypes)
        # Yorg1 = Yorg[Yorg['Outcome'] == 1]
        # Yorg2 = Yorg[Yorg['Outcome'] == 2]
        # Yorg3 = Yorg[Yorg['Outcome'] == 3]
        # print("Yorg", Yorg.head())
        # print("Pred", data['Pred'].head())
        # print("Correct", data['Correct'].head())
        ##print(Yorg.loc[3])
        # print("SBRESULT : Yorg1={} Yorg2={} Yorg3={}".format(len(Yorg1), len(Yorg2), len(Yorg3)))

        # Correct1 = 1.0*((data['Pred'] == 1)and(data['Outcome'] == 1))
        # data[Correct1] = 1.0*(data['Pred'] == data['Outcome'])
        # print("len Pred", len(data['Pred']))
        # print("len Outcome", len(data['Outcome']))
        # print("SBRESULT : Correct1={} Correct2={} Correct3={}".format(len(Correct1), len(Correct1), len(data['Correct'])))

        # wrong = data[data['Correct'] == 0.0]
        # print(wrong.head())

        """
        #%%
        # Let's denote on the figure which ones we got wrong
        wrong = data[data['Correct'] == 0.0]
        plt.scatter(wrong[col1],
                    wrong[col2],
                    marker = 's',
                    facecolors='none',
                    edgecolors= 'k',
                    s = 100)

        plt.scatter(wrong[col1],
            wrong[col2],
            marker = 'x',
            facecolors='k',
            edgecolors= 'k',
            s = 100)

        """

        """
        #%%
        # It is also possible to predict probabilities
        data['Probs'] = LR.predict_proba(X)[:,1]

        # We can show that the prediction gets stronger further away
        # from the decision boundary
        plt.figure()
        plt.scatter(X[col1],
                    X[col2],
                    c = data['Probs'],#.to_numpy().reshape(N).tolist(),
                    s = 100)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.colorbar()

        # Plot the line
        plt.plot([dim1_min,dim1_max],
                [y_dim1_min,y_dim1_max],
                c='k')
        plt.xlim([dim1_min, dim1_max + 1])
        plt.ylim([0, dim2_max + 1])

        """

        # break
    # break



