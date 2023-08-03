# Created by Jamilah Foucher, May 09, 2021 

# Purpose: to quickly use the scikit-learn python machine learning toolbox.  The most frequently used regression and classification methods are outlined such that they can be quickly used.

# ----------------------------------------------
# EDA automatic
# ----------------------------------------------
pip install pandas-profiling
import pandas as pd
from pandas_profiling import ProfileReport

#EDA using pandas-profiling
profile = ProfileReport(pd.read_csv('titanic.csv'), explorative=True)

#Saving results to a HTML file
profile.to_file("output.html")

# ---------------------

pip install autoviz
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class

#EDA using Autoviz
autoviz = AutoViz_Class().AutoViz('train.csv')

# ---------------------

pip install sweetviz
import pandas as pd
import sweetviz as sv

#EDA using Autoviz
sweet_report = sv.analyze(pd.read_csv("titanic.csv"))

#Saving results to HTML file
sweet_report.show_html('sweet_report.html')

# ---------------------

pip install dtale
import dtale
import pandas as pd

dtale.show(pd.read_csv("titanic.csv"))

# ---------------------


# ---------------------


# ----------------------------------------------
# Pandas tips
# ----------------------------------------------
X = df.drop(['label'], axis=1)
y = df['label']

# ---------------------

# Drop plusieurs columns
X = df.drop('bmi', axis=1).drop('device_id', axis=1)

# ---------------------

# Créer des valueur nan dans un DataFrame pour un column specifique
df.loc[df.sample(frac=0.18).index, 'min_active_heartrate'] = np.nan
df.loc[df.sample(frac=0.05).index, 'min_steps'] = np.nan

# ---------------------

# Convert numerical strings à des valeurs numerique
# Si il y a des entries type string comme '2' par rapport 2, pd.to_numeric convertera '2' à 2. apply appellera le function pd.to_numeric et execute le fonction pour le Dataframe df0
df0 = df0.apply(pd.to_numeric)

# ---------------------


# ---------------------



# ----------------------------------------------

# ----------------------------------------------
# Pre-treating features or labels
# ----------------------------------------------

# Handling missing values Steps
# ---------------------
# Regard à le donnees
# ---------------------
# 0. Verifier la somme de nan par column
X_test.isnull().sum()

# 1. Glimpse the dataset and calculate non-null values offline
print(f"The number of rows in this dataset is {ht_lifestyle_pd_df.shape[0]}")
ht_lifestyle_pd_df.info()

# 2. Output all the dataframe columns are have null values
ht_lifestyle_pd_df.loc[:, ht_lifestyle_pd_df.isnull().any()]

# ---------------------
# Fillin nan avec (mean, median, mode) --> median est mieux parce que c'est influence la moyen moins
# ---------------------
X_train['avg_resting_heartrate'] = X_train['avg_resting_heartrate'].fillna(avg_rest_heartrate_median)

# Save missing value locations to a variable for indexing
missing_values_train = X_train[X_train['avg_resting_heartrate'].isnull() == True].index
missing_values_test = X_test[X_test['avg_resting_heartrate'].isnull() == True].index
# Then using loc and the index, find all of the rows in the avg_resting_heartrate column and fill
X_train.loc[missing_values_train, 'avg_resting_heartrate'] = avg_rest_heartrate_median
X_test.loc[missing_values_test, 'avg_resting_heartrate'] = avg_rest_heartrate_median


# ---------------------
# One-hot encoding
# ---------------------

# 0. Trouver un column categorical que tu veux transformer en column numerique
ht_lifestyle_pd_df.dtypes
ht_lifestyle_pd_df.info()

# 1. One-hot encode
import pandas as pd
lifestyle_dummies_df = pd.get_dummies(ht_lifestyle_pd_df['lifestyle'])
ht_lifestyle_pd_df = ht_lifestyle_pd_df.join(lifestyle_dummies_df)
ht_lifestyle_pd_df.drop('lifestyle', axis=1, inplace=True)

# OU

ht_lifestyle_pd_df = pd.get_dummies(ht_lifestyle_pd_df, prefix='ohe', columns=['lifestyle'])
# ---------------------



# ---------------------
# Categorical encoding
# ---------------------
df['lifestyle'] = df['lifestyle'].map({'Sedentary':0, 'Weight Trainer':1, 'Athlete':1, 'Cardio Enthusiast':1})

# OU

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_numeric'] = le.fit_transform(df['label'])


# ---------------------

# ----------------------------------------------


# ----------------------------------------------
# Fill missing values automatically

def fill_missing_values(X, filltype='mean'):
  
  col = X.columns
  
  # ----------------------
  # AVANT: Verifier qu'il n'y a pas des valeurs nan
  out = X.isnull().sum()
  dd = dict(zip(list(out.index), list(out)))
  x = list(filter(lambda x: x[1] != 0, dd.items()))
  print('AVANT dd: ', dd)
  # ----------------------
  
  # ----------------------
  for i in x:
    
    if filltype == 'mean':
      fill_val = X[i[0]].mean()
    elif filltype == 'median':
      fill_val = X[i[0]].median()
    elif filltype == 'mode':
      fill_val = X[i[0]].mode()
    elif filltype == 'min':
      fill_val = X[i[0]].min()
    elif filltype == 'max':
      fill_val = X[i[0]].max()
    # ----------------------
    print('Fill value is ', filltype, 'of ', i[0], ': ', f"{fill_val:.2f}")
    
    # X[i[0]].fillna(fill_val)

    # OU si fillna ne marche pas

    missing_values = X[X[i[0]].isnull() == True].index
    X.loc[missing_values, i[0]] = fill_val
  # ----------------------
    
  # ----------------------
  # APRES: Verifier qu'il n'y a pas des valeurs nan
  out = X.isnull().sum()
  dd = dict(zip(list(out.index), list(out)))
  print('APRES dd: ', dd)
  # ----------------------
  
  X = pd.DataFrame(X)
  X.columns = col
  
  return X


X_train = fill_missing_values(X_train, filltype='mean')
X_test = fill_missing_values(X_test, filltype='mean')

# ----------------------------------------------







from mlxtend.preprocessing import minmax_scaling

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
import numpy as np

def scale_feature_data(feat, plotORnot):
    
    columns = ['0']
    dat = pd.DataFrame(data=feat, columns=columns)
    scaled_data0 = minmax_scaling(dat, columns=columns)
    scaled_data_mlx = list(scaled_data0.to_numpy().ravel())
    # OR 
    scaled_data_norma = []
    for q in range(len(feat)):
        scaled_data_norma.append( (feat[q] - np.min(feat))/(np.max(feat) - np.min(feat)) )  # normalization : same as mlxtend
    # OR 
    shift_up = [i - np.min(feat) for i in feat]
    scaled_data_posnorma = [q/np.max(shift_up) for q in shift_up]  # positive normalization : same as mlxtend
    # OR 
    scaled_data_standardization = [(q - np.mean(feat))/np.std(feat) for q in feat]  # standardization
    
    if plotORnot == 1:
        fig = make_subplots(rows=2, cols=1)
        config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})
        xxORG = list(range(len(feat)))
        fig.add_trace(go.Scatter(x=xxORG, y=feat, name='feat', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_mlx, name='scaled : mlxtend', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_norma, name='scaled : normalization', line = dict(color='cyan', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_posnorma, name='scaled : positive normalization', line = dict(color='blue', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxORG, y=scaled_data_standardization, name='scaled : standardization', line = dict(color='orange', width=2, dash='solid'), showlegend=True), row=2, col=1)
        fig.update_layout(title='feature vs scaled featue', xaxis_title='data points', yaxis_title='amplitude')
        fig.show(config=config)

    return scaled_data_mlx

# ----------------------------------------------

def train_validation_test_split():
    from sklearn.model_selection import train_test_split
    
    train_val_df, test_df = train_test_split(ht_user_metrics_pd_df, train_size=0.85, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, train_size=0.7, test_size=0.3, random_state=42)
    
# ----------------------------------------------

def grid_search():
    
    # max_depth - the maximum depth of each tree
    # n_estimators – the number of trees in the forest
    parameter_grid = {
      'max_depth':[2, 4, 5, 8, 10, 15, 20, 25, 30], 
      'n_estimators':[3, 5, 10, 50, 100, 150, 250, 500]
    }
    
    # .....................................
    # total unique combinations of hyperparameters are there in parameter_grid
    lenn = [len(i) for i in parameter_grid.values()]
    print('lenn: ', lenn)
    mult_tot = lenn[0]
    for i in range(1,len(lenn)):
      mult_tot = mult_tot*lenn[i]
    print('mult_tot: ', mult_tot)
    # .....................................
        
    # Predefined Split: create a predefined split to pass into a our grid-search process.
    from sklearn.model_selection import PredefinedSplit

    # Create list of -1s for training set row or 0s for validation set row
    split_index = [-1 if row in train_df.index else 0 for row in train_val_df.index]

    # Create predefined split object
    predefined_split = PredefinedSplit(test_fold=split_index)
    
    # -------------------------------
    
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(estimator=rfc, cv=predefined_split, param_grid=parameter_grid)
    
    # Training the Models
    grid_search.fit(train_val_df.drop("steps_10000", axis=1), train_val_df["steps_10000"])
    
    # -------------------------------
    
    # {'max_depth': 15, 'n_estimators': 100}
    grid_search.best_params_
    
    grid_search.best_score_
    
    # Obtenez la prediction
    y_pred = grid_search.predict(test_df.drop("steps_10000", axis=1)
    
    from sklearn.metrics import accuracy_score
    accuracy_score(y, y_pred)
    
    # -------------------------------
    
    # Cross-validated Results
    # If you want to examine the results for each individual fold, you can use grid_search's cv_results_ attribute.
    # Note that each row of the DataFrame corresponds to a unique set of hyperparameters.
    import pandas as pd
    pd.DataFrame(grid_search.cv_results_).head()


# ----------------------------------------------

def do_train_test_split(X, y):
    
    from sklearn.model_selection import train_test_split
    import numpy as np
    seed = 0
    X_train, X_test, Y_train_1D, Y_test_1D = train_test_split(X, y, train_size=0.9, test_size=0.10, random_state = seed)
    
    # OU Si pandas
    if type(X_train) == 'pandas.core.frame.DataFrame':
        X_train.reset_index(drop=True,inplace=True)
        X_test.reset_index(drop=True,inplace=True)
        Y_train.reset_index(drop=True,inplace=True)
        Y_test.reset_index(drop=True,inplace=True)
    else:
        # Make numpy
        X_train = np.array(X_train)
        print('shape of X_train : ', X_train.shape)

        Y_train_1D = np.array(Y_train_1D)
        print('shape of Y_train_1D : ', Y_train_1D.shape)

        X_test = np.array(X_test)
        print('shape of X_test : ', X_test.shape)

        Y_test_1D = np.array(Y_test_1D)
        print('shape of Y_test_1D : ', Y_test_1D.shape)
    
    return X_train, X_test, Y_train_1D, Y_test_1D
# ----------------------------------------------



# ----------------------------------------------
def check_if_Y_1D_is_correct(Y_1D):
    
    import matplotlib.pyplot as plt
    fig, (ax0) = plt.subplots(1)

    ax0.plot(Y_1D[:], 'b-', label='Y_1D')
    ax0.set_ylabel('Y_1D')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return
# ----------------------------------------------


# ----------------------------------------------
def binarize_Y1Dvec_2_Ybin(Y):
    
    import numpy as np
    
    # Transform a 1D Y vector (n_samples by 1) to a Y_bin (n_samples by n_classes) vector

    # Ensure vector is of integers
    Y = [int(i) for i in Y]

    # Number of samples
    m_examples = len(Y)

    # Number of classes
    temp = np.unique(Y)
    unique_classes = [int(i) for i in temp]
    # print('unique_classes : ', unique_classes)

    whichone = 2
    # Binarize the output
    if whichone == 0:
        from sklearn.preprocessing import label_binarize
        Y_bin = label_binarize(Y, classes=unique_classes)  # does not work

    elif whichone == 1:
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        Y_bin = lb.fit_transform(Y)
        
    elif whichone == 2:
        # By hand
        Y_bin = np.zeros((m_examples, len(unique_classes)))
        for i in range(0, m_examples):
            if Y[i] == unique_classes[0]:
                Y_bin[i,0] = 1
            elif Y[i] == unique_classes[1]:
                Y_bin[i,1] = 1
            elif Y[i] == unique_classes[2]:
                Y_bin[i,2] = 1
            elif Y[i] == unique_classes[3]:
                Y_bin[i,3] = 1
            elif Y[i] == unique_classes[4]:
                Y_bin[i,4] = 1
            elif Y[i] == unique_classes[5]:
                Y_bin[i,5] = 1
            elif Y[i] == unique_classes[6]:
                Y_bin[i,6] = 1
                
    print('shape of Y_bin : ', Y_bin.shape)

    return Y_bin, unique_classes
# ----------------------------------------------


# ----------------------------------------------
def debinarize_Ybin_2_Y1Dvec(Y_bin):

    import numpy as np

    # Transform a Y_bin (n_samples by n_classes) vector to a 1D Y vector (n_samples by 1)

    # De-Binarize the output
    Y = np.argmax(Y_bin, axis=1)

    return Y
# ----------------------------------------------


# ----------------------------------------------
def transform_Y_bin_pp_2_Y_1D_pp(Y_1D, Y_bin_pp):

    # Y_bin_pp is size [n_samples, n_classes=2]
    # Take the column of Y_bin_pp for the class of Y_1D, because both vectors need to be [n_samples, 1]
    import numpy as np
    Y_1D_pp = []
    for q in range(len(Y_1D)):
        desrow = Y_bin_pp[q]
        Y_1D_pp.append(desrow[int(Y_1D[q])])
    Y_1D_pp = np.ravel(Y_1D_pp)
    
    Y_1D_pp = np.array(Y_1D_pp)
    
    return Y_1D_pp

# ----------------------------------------------


# ----------------------------------------------
# Summarized Preprocessing and model fitting: Pipeline
# ----------------------------------------------

def modele_processus(X, y, model):
  
  faitOUpas = 0
  if faitOUpas == 1:
    col = X.columns
    from sklearn.preprocessing import MinMaxScaler
    scaled_features = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(scaled_features)
    X.columns = col
  
  from sklearn.model_selection import train_test_split
  X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42) 
  
  faitOUpas = 0
  if faitOUpas == 1:
    X_train = fill_missing_values(X_train, filltype='mean')
    X_test = fill_missing_values(X_test, filltype='mean')
  
  # ----------------------------------
  print('X_train.shape: ', X_train.shape)
  print('X_test.shape: ', X_test.shape)
  # ----------------------------------
  
  model.fit(X_train, Y_train)

  # ----------------------------------
  laquelle = 1
  if laquelle == 0:
    from sklearn.metrics import r2_score
    y_train_predicted = model.predict(X_train)
    y_test_predicted = model.predict(X_test)
    rsquared_train = r2_score(Y_train, y_train_predicted)
    rsquared_test = r2_score(Y_test, y_test_predicted)
    print("R2 on training set: ", round(rsquared_train, 3) )
    print("R2 on test set: ", round(rsquared_test, 3) )
    
  elif laquelle == 1:
    rsquared_train = model.score(X_train, Y_train)
    rsquared_test = model.score(X_test, Y_test)
    print("R2 on training set: ", round(rsquared_train, 3) )
    print("R2 on test set: ", round(rsquared_test, 3) )
    
  elif laquelle == 2:
    from sklearn.metrics import accuracy_score
    # Train accuracy
    train_accuracy = accuracy_score(train_df["steps_10000"], rfc.predict(train_df.drop("steps_10000", axis=1)))

    # Test accuracy
    test_accuracy = accuracy_score(test_df["steps_10000"], rfc.predict(test_df.drop("steps_10000", axis=1)))
    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)
  # ----------------------------------

  
  
  return X_train, X_test, Y_train, Y_test, model
  
# ----------------------------------------------





# ----------------------------------------------
# Clustering
# ----------------------------------------------

def plot_feature_space_2D(X, y):
    
    import matplotlib.pyplot as plt
    
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Plot the first two features
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    
    return
# ----------------------------------------------


# ----------------------------------------------
def plot_feature_space_3D_PCA(X, y):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    
    # To get a better understanding of interaction of the dimensions plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    
    # PCA of the first two features
    X_reduced = PCA(n_components=3).fit_transform(X)

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="k", s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()
    
    return
    
# ----------------------------------------------

# Principle Component Analysis (PCA)
def get_pca(X_train, n_components):

    # X_train is size [n_samples, n_features]
    # n_components is a scalar, the number of dominate eigenvectors/components that make up the data
    
    # ...............................
    # Normalize the numeric features so they're on the same scale
    from sklearn.preprocessing import MinMaxScaler
    scaled_features = MinMaxScaler().fit_transform(X_train)
    # OU
    from sklearn.preprocessing import StandardScaler
    scaled_features = StandardScaler().fit_transform(X_train)
    # OU
    from sklearn.preprocessing import scale
    scaled_features = scale(X_train)
    # ...............................

    from sklearn.decomposition import PCA
    
    # Get the n_components principle componets of the data X_train
    X_PCA = PCA(n_components=n_components, random_state=42).fit_transform(scaled_features)
    # OU
    PCA_fit = PCA(n_components=n_components).fit(scaled_features)  # the principal component model
    X_PCA = PCA_fit.transform(scaled_features) # apply the dimensionality reduction on X
    
    import matplotlib.pyplot as plt
    plt.plot(X_PCA[:])
    plt.title('PCA')
    plt.xlabel('samples')
    plt.ylabel('magnitude')
    plt.show()
    
    # ...............................
    
    # Print the number of components
    print('n_components: ', PCA_fit.n_components_)
    
    # ...............................
    
    # Explained Variance
    e_var = pca.explained_variance_ratio_
    
    import matplotlib.pyplot as plt
    import numpy as np

    plt.bar(range(1, 21), pca.explained_variance_ratio_) 
    plt.xlabel('Component') 
    plt.xticks(range(1, 21))
    plt.ylabel('Percent of variance explained')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    
    # ...............................
    
    plt.plot(range(1, 21), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Component') 
    plt.xticks(range(1, 21))
    plt.ylabel('Percent of cumulative variance explained')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()
    
    # ...............................
    
    return X_PCA

# ----------------------------------------------


# ------------------------------
# Process for PCA analysis
# ------------------------------
# 0. Executer PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA(random_state=42)
pca.fit(scale(df))

# 1. Identifier la quelle component expliquer les quantities des variances
import matplotlib.pyplot as plt
import numpy as np

plt.bar(range(1, 25), pca.explained_variance_ratio_) 
plt.xlabel('Component') 
plt.xticks(range(1, 25))
plt.ylabel('Percent of variance explained')
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

print("percentage des variances: ", pca.explained_variance_ratio_)

# 2. Quanities des components qui expliquer 90% des variances
plt.plot(range(1, 25), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Component') 
plt.xticks(range(1, 25))
plt.ylabel('Percent of cumulative variance explained')
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

for component in list(zip(range(1, 25), np.cumsum(pca.explained_variance_ratio_))):
  if component[1] >= 0.9:
    print(component)
    break


# 3. Compute la matrice de loading comprendre la quelle component sont correlater avec des features
#  Loadings are the unstandardized eigenvectors' elements, i.e. eigenvectors endowed by corresponding component variances, or eigenvalues.

#  The singular values of a M×N matrix X are the square roots of the eigenvalues of the N×N matrix X∗X (where ∗ stands for the transpose-conjugate matrix if it has complex coefficients, or the transpose if it has real coefficients). 

# loadings = pca.components_.T * np.sqrt(pca.explained_variance_) 
# OR
loadings = pca.components_.T * np.sqrt(pca.singular_values_**2)

component_columns = ["PC" + str(x) for x in range(1, 25)]
loadings_df = pd.DataFrame(loadings, columns=component_columns, index=df.columns)
loadings_df

# 4. Determiner le quel component sont correlate avec quel feature
# (ie: Le premiere component PC1 est plus correlate avec la feature avg_steps)
abs(loadings_df["PC1"]).sort_values(ascending=False)

# 5. Selectionner des components pour la nouvelle matrices des features (x)
component_df = pd.DataFrame(pca.transform(scale(df)), columns=component_columns)
X = component_df.loc[:, ["PC1", "PC2", "PC3"]]
X


# ----------------------------------------------

# Canonical Correlation Analysis (CCA)
def get_cca(X_train, Y_train, n_components):

    # X_train is size [n_samples, n_features]
    # Y_train is size (n_samples, n_classes)
    # n_components is a scalar, the number of dominate components that make up the data
    
    # ...............................
    # Normalize the numeric features so they're on the same scale
    from sklearn.preprocessing import MinMaxScaler
    scaled_features = MinMaxScaler().fit_transform(X_train)
    # OU
    from sklearn.preprocessing import StandardScaler
    scaled_features = StandardScaler().fit_transform(X_train)
    # OU
    from sklearn.preprocessing import scale
    scaled_features = scale(X_train)
    # ...............................
    
    from sklearn.cross_decomposition import CCA

    # Get the n_components principle componets of the data X_train
    X_CCA = CCA(n_components=n_components).fit(scaled_features, Y_train).transform(scaled_features)
    
    import matplotlib.pyplot as plt
    plt.plot(X_CCA[:])
    plt.set_title('CCA')
    plt.set_xlabel('samples')
    plt.set_ylabel('magnitude')
    plt.show()

    return X_CCA

# ----------------------------------------------

def unsupervised_lab_kmeans_clustering(*arg):
    
    from sklearn.cluster import KMeans
    n_clusters = arg[0]
    X = arg[1]
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',algorithm='elkan', random_state=2)
    # n_clusters : The number of clusters to form as well as the number of centroids to generate. (int, default=8)
    
    # init : Method for initialization : (default=’k-means++’)
    # init='k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. 
    # init='random': choose n_clusters observations (rows) at random from data for the initial centroids.
    
    # n_init : Number of time the k-means algorithm will be run with different centroid seeds (int, default=10)
    
    # max_iter : Maximum number of iterations of the k-means algorithm for a single run. (int, default=300)
    
    # tol : Relative tolerance with regards to Frobenius norm of the difference in the cluster centers 
    # of two consecutive iterations to declare convergence. (float, default=1e-4)
    
    # (extremly important!) random_state : Determines random number generation for centroid initialization
    #(int, RandomState instance or None, default=None)
    
    # algorithm{“auto”, “full”, “elkan”}, default=”auto”
    # K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more 
    # efficient on data with well-defined clusters, by using the triangle inequality. However it’s more 
    # memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
    
    # ------------------------------
    
    # print('shape of X : ', X.shape)
    kmeans.fit(X)

    # ------------------------------

    # Get the prediction of each category : predicted label
    label = kmeans.labels_
    # print('clusters_out : ' + str(clusters_out))
    # OR
    label = kmeans.predict(X)
    # print('clusters_out : ' + str(clusters_out))
    # print('length of clusters_out', len(clusters_out))
    
    # ------------------------------
    
    # Centroid values for feature space : this is the center cluster value per feature in X
    centroids = kmeans.cluster_centers_
    # print('centroids org : ' + str(centroids))
    
    # ------------------------------
    
    return kmeans, label, centroids
    
    
# ----------------------------------------------


# ------------------------------
# Process for unsupervised training and testing/inference
# ------------------------------
# Find the optimal number of clusters: Elbow method
elbow_method(X)

# Determine best number of clusters
n_cluster = 4

# Split data into train and test
train_df, inference_df = train_test_split(X, train_size=0.9, test_size=0.1, random_state=42)

# Train model
k_means, label, centroids = unsupervised_lab_kmeans_clustering(n_cluster, train_df)

# Test model with test/inference dataset
inference_df_clusters = k_means.predict(scale(inference_df))

# copy the test/inference dataset
clusters_df = inference_df.copy()

# Add the label/cluster prediction to the copied dataset
clusters_df["cluster"] = inference_df_clusters

# Display dataframe
display(clusters_df)
    

# ----------------------------------------------

def elbow_method(X):
    # Kmeans: Elbow method (use a distance measure to quantify distance between data points and clusters)
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    %matplotlib inline

    # ...............................
    # Normalize the numeric features so they're on the same scale
    from sklearn.preprocessing import MinMaxScaler
    scaled_features = MinMaxScaler().fit_transform(X)
    # OU
    from sklearn.preprocessing import StandardScaler
    scaled_features = StandardScaler().fit_transform(X)
    # OU
    from sklearn.preprocessing import scale
    scaled_features = scale(X)
    # ...............................


    laquelle = 1

    if laquelle == 0:
        # Create 10 models with 1 to 10 clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i)
            
            # Fit the data points
            kmeans.fit(scaled_features)
            
            # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided
            wcss.append(kmeans.inertia_)
            
        #Plot the WCSS values onto a line graph
        plt.plot(range(1, 11), wcss)
        plt.title('WCSS by Clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        
    elif laquelle == 1:
        # Utiliser scaled 2-norm et Bayesian information criterion (BIC) pour un measure des distances entre des points
        bic_tot = []
        # sil_val = []
        # hom_val = []
        for n_clusters in range(2, 15):
            model, y, centroids = unsupervised_lab_kmeans_clustering(n_clusters, scaled_features)
            # ----------------------------

            # sil_val.append(metrics.silhouette_score(scaled_features, y))
            # hom_val.append(metrics.homogeneity_score(scaled_features, y))

            # Compute Bayesian information criterion (BIC) : existing function - mais, c'est prend beaucoup du temps
            # model = GaussianMixture(n_components=n_clusters)
            # y_gm = model.fit_predict(scaled_features)
            # bic_tot.append(model.bic(scaled_features))
            # aic = model.aic(scaled_features) # Akaike information criterion (AIC) 

            # ----------------------------

            # Compute BIC : by hand

            # number of elements in each cluster
            vals, cnt = np.unique(y, return_counts=True)

            r, c = scaled_features.shape

            # compute variance of clusters
            cl_var = [(1/(cnt[i]-n_clusters))*np.linalg.norm(scaled_features[np.where(y == i)]-centroids[i][:], ord=2) for i in range(n_clusters)]
            # print('cl_var : ', cl_var)

            const_term = 0.5 * n_clusters * np.log(r) * (c+1)

            n = np.bincount(y)
            BIC = np.sum([n[i] * np.log(n[i]) -
                       n[i] * np.log(r) -
                     ((n[i] * c) / 2) * np.log(2*np.pi*cl_var[i]) -
                     ((n[i] - 1) * c/ 2) for i in range(n_clusters)]) - const_term
            bic_tot.append(BIC)

        # ----------------------------

        fig, (ax0, ax1) = plt.subplots(2)
        # plotting variance
        out = [np.log(2*np.pi*cl_var[i]) for i in range(len(cl_var))]
        ax0.plot(np.arange(2, 2+len(out)), out)
        ax0.set_ylabel('Variance')

        # plot clusters by BIC
        ax1.plot(np.arange(2,2+len(bic_tot)), bic_tot)
        ax1.set_ylabel('BIC score')
        ax1.set_xlabel('n_clusters')
        
    elif laquelle == 2:
        from sklearn.preprocessing import scale
        # Utiliser RMSE (which is the 2-norm normalized) pour un measure des distances entre des points
        distortions = []
        values_of_k = range(2, 10)
        
        for k in values_of_k:
            k_means = KMeans(n_clusters=k, random_state=42)
            k_means.fit(scale(df))
            distortion = k_means.score(scale(df))  # score gives negative r-squared value, so invert it to create a proper plot
            distortions.append(-distortion)
          
        import matplotlib.pyplot as plt
        plt.plot(values_of_k, distortions, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Distortion') 
        plt.show()
        
    return

# ----------------------------------------------



# ----------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)

# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(scaled_features)

# View the cluster assignments
km_clusters

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)

# ----------------------------------------------

# Hierarchical clustering methods make fewer distributional assumptions when compared to K-means methods. However, K-means methods are generally more scalable, sometimes very much so.

# Hierarchical clustering creates clusters by either a divisive method or agglomerative method. The divisive method is a "top down" approach starting with the entire dataset and then finding partitions in a stepwise manner. Agglomerative clustering is a "bottom up** approach. 
from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
agg_clusters
import matplotlib.pyplot as plt

%matplotlib inline

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, agg_clusters)


# ----------------------------------------------


# Clustering evaluation : 







# ----------------------------------------------
# Machine Learning Classification :
# ----------------------------------------------
# multi-class Stochastic Gradient Descent
def multiclass_stochastic_gradient_descent(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size (n_samples, n_features)
    # X_test is size (n_samples, n_features)
    # Y_train_1D is size (n_samples, 1) where the n_classes are represented by 1,2,3,..., etc
    
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    # Can ONLY calculate the probability estimates (predict_proba), but NOT the decision_function
    lr = SGDClassifier(loss='hinge', alpha=0.001, class_weight='balanced')
    clf =lr.fit(X_train, Y_train_1D)
    calibrator = CalibratedClassifierCV(clf, cv='prefit')
    model = calibrator.fit(X_train, Y_train_1D)
    
    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # probability estimates (predict_proba) are not available for loss='hinge', but can calculate the decision_function
    model = SGDClassifier(alpha=0.001, loss='hinge') 
    model.fit(X_train, Y_train_1D)
    
    # -------
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #  size is [n_samples, n_classes]
    Y_train_bin_score = model.decision_function(X_train)
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    # -------

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Linear Discriminant Analysis (LDA)
def multiclass_LDA_classifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis()

    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------

    # -------
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #  size is [n_samples, n_classes]
    Y_train_bin_score = model.decision_function(X_train)
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    # -------

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Support Vector Machine, also known as C-Support Vector Classifier, (SVC)
def multiclass_svm(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier
    
    # class_weight=balanced: Automatically adjust weights inversely proportional to class frequencies in the input data
    # C parameter: Penalty parameter

    model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, class_weight=None, max_iter=100))  #, random_state=random_state
    # OR
    # model = svm.SVC(decision_function_shape='ovr', probability=True, max_iter=100)  # “one-versus-rest” : multi-class (need to try **)
    # Y_train_bin is (n_samples, n_classes), this transforms the ovo setup into a ovr setup
    # OR
    # model = svm.SVC(kernel='linear', class_weight="balance", c=1.0, random_state=0)
    # OR
    # model = svm.LinearSVC() # “one-versus-rest” : multi-class (need to try **) Y_train_bin is (n_samples, n_classes), direct ovr setup
    
    model.fit(X_train, Y_train_bin)
    
    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    #print('Y_test_bin_predict : ', Y_test_bin_predict)

    # The prediction probability of each class
    Y_train_bin_pp = model.predict_proba(X_train) # size is [n_samples, n_classes]
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)

    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_bin_score = model.decision_function(X_train)  # size is n_samples, 1
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    # Y_train_bin and Y_test_bin needs to be used in the evaluation of the model ALSO!!
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Support Vector Machine (NuSVC)
def multiclass_svm_NuSVC(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn import svm
    
    model = svm.NuSVC(decision_function_shape='ovr', probability=True, max_iter=1) 
    #  class_weight='auto',
    # “one-versus-rest” : multi-class Y_train_bin is (n_samples, n_classes), direct ovr setup
    
    model.fit(X_train, Y_train_1D)
    
    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------

    # -------
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #  size is [n_samples, n_classes]
    Y_train_bin_score = model.decision_function(X_train)
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    # -------
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Faster Multi-class Support Vector Machine (SVC) : Bagging
def multiclass_svm_bagging(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC

    n_estimators = 10
    #max_samples= 1.0 / n_estimators  # max_samples must be in (0, n_samples]
    #print('max_samples : ', max_samples)
    #model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight=None, max_iter=100), n_estimators=n_estimators))

    model.fit(X_train, Y_train_bin)

    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    #print('Y_test_bin_predict : ', Y_test_bin_predict)

    # The prediction probability of each class
    Y_train_bin_pp = model.predict_proba(X_train) # size is [n_samples, n_classes]
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)

    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    Y_train_bin_score = model.decision_function(X_train)  # size is n_samples, 1
    Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    # Y_train_bin and Y_test_bin needs to be used in the evaluation of the model ALSO!!
    
    #score(X, y[, sample_weight]) Return the mean accuracy on the given test data and labels.
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# 2-class RandomForest
def binary_RandomForest(*args):
    
    X_train = args[0]
    X_test = args[1]
    Y_train_1D = args[2]
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(random_state=42)
    # OU
    model = RandomForestClassifier(random_state=1, min_samples_leaf=50)  # min_samples_leaf is 100 by default
    
    # Regardez à des hyperparametres
    rfc.get_params()
    
    # Changez des valeures des hyperparametres
    rfc.set_params(max_depth=2, n_estimators=3)
    
    Y_train_1D = np.reshape(Y_train_1D, (len(Y_train_1D), 1))  # Y needs to have a defined shape ***
    model.fit(X_train, Y_train_1D)

    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) 
    Y_test_bin_pp = model.predict_proba(X_test)
    
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), 2))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), 2))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    if len(args) > 3:
        Y_test_1D = args[3]
        Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
        Y_test_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_test_1D, Y_test_bin_pp)
        # OR
        # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
        #Y_train_1D_score = model.decision_function(X_train)  # size is [n_samples, 1]
        #Y_test_1D_score = model.decision_function(X_test)
        
        Y_train_1D_score = np.array(Y_train_1D_score)
        print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
        Y_test_1D_score = np.array(Y_test_1D_score)
        print('shape of Y_test_1D_score : ', Y_test_1D_score.shape)
    else:
        Y_train_1D_score = transform_Y_bin_pp_2_Y_1D_pp(Y_train_1D, Y_train_bin_pp)
        Y_train_1D_score = np.array(Y_train_1D_score)
        print('shape of Y_train_1D_score : ', Y_train_1D_score.shape)
        Y_test_1D_score = []

    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_1D_score, Y_test_1D_score



# ----------------------------------------------

# Multi-class RandomForest
def multiclass_RandomForest_1Dinput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # -------
    
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(random_state=1, min_samples_leaf=50)  # min_samples_leaf is 100 by default
    model = MultiOutputClassifier(forest, n_jobs=-1) #n_jobs=-1 means apply parallel processing
    
    Y_train_1D = np.reshape(Y_train_1D, (len(Y_train_1D), 1))  # Y needs to have a defined shape ***
    model.fit(X_train, Y_train_1D)
    
    
    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [1, n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    unique_classes = np.unique(Y_train_1D)
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    
    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# Multi-class RandomForest : is the same as the 1Dinput
def multiclass_RandomForest_bininput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(random_state=1, min_samples_leaf=50)  # min_samples_leaf is 100 by default
    model = MultiOutputClassifier(forest, n_jobs=-1) #n_jobs=-1 means apply parallel processing
    
    model.fit(X_train, Y_train_bin)
    
    # -------
    # Y_predict : size [n_samples, n_classes]
    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    
    Y_train_bin_predict = np.array(Y_train_bin_predict)
    print('shape of Y_train_bin_predict : ', Y_train_bin_predict.shape)
    Y_test_bin_predict = np.array(Y_test_bin_predict)
    print('shape of Y_test_bin_predict : ', Y_test_bin_predict.shape)
    # -------
    
    
    # Can not use the same model with Y_train_bin, as the prediction probability is per class instead of across class
    Y_train_1D = np.reshape(Y_train_1D, (len(Y_train_1D), 1))  # Y needs to have a defined shape ***
    model.fit(X_train, Y_train_1D)
    
    # -------
    # The prediction probability of each class : size is [1, n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    unique_classes = np.unique(Y_train_1D)
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    
    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# ----------------
# XGBoost
# ----------------
# Ensemble methods : they combine several precise predictive models (each model predicts one thing separately) in order to predict across the total prediction items - trees and random forest is an ensemble method, as each leaf is a separate model.
 
# Gradient Boosting : a method that goes through cycles to iteratively add models into an ensemble - the process is (for each model) : train model, test predictions, calculate loss (mean squared error, accuracy), improve model (train new model - "use the loss function to fit a new model"), add new model to ensemble

# The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)

# XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features focused on performance and speed
# ----------------------------------------------

# Multi-class XGBoost RandomForest 
# written in an more efficient manner, so it is faster and more accurate than RandomForest.
def multiclass_XGBClassifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from xgboost import XGBClassifier
    
    model = XGBClassifier(num_iterations=1000, eval_metric='mlogloss', boosting='gbdt')
    # can not say num_class=2, gives error
    #num_class=2, learning_rate=0.1,  max_depth=10, feature_fraction=0.7, 
    #scale_pos_weight=1.5, boosting='gbdt', metric='multiclass')
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Gradient Boosting Classifier (gradient descent w/ logistic regression cost function)
def multiclass_GradientBoostingClassifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0, n_iter_no_change=10)
    
    # loss{‘deviance’, ‘exponential’}, default=’deviance’ : The loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.

    # max_depth=1, 
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# Multi-class Decision Tree Classifier
def multiclass_Decision_Tree_Classifier(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------

    # -------
    # The prediction probability of each class : size is [1, n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    unique_classes = np.unique(Y_train_1D)
    Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------

    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    
    #tree.plot_tree(model)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score


# ----------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)


# ----------------------------------------------

# NearestNeighbors
def Nearest_Neighbors(X_train, Y_train_1D, X_test):
    weights = 'uniform' # 'distance'
    n_neighbors = 15
    model = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    model.fit(X_train, Y_train_1D)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
    h = 0.02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  #decision boundary

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("n-Class classification: training (k = %i, weights = '%s')" % (n_neighbors, weights))

    plt.show()
    
    return model, Z

# ----------------------------------------------

# Multilayer perceptron (MLP)/neural network (Deep Learning) : logistic regression NN
def multiclass_multilayer_perceptron_bininput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_train_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_train_1D)
    Y_test_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_test_1D)

    Y_train_bin = np.array(Y_train_bin)
    print('shape of Y_train_bin : ', Y_train_bin.shape)

    Y_test_bin = np.array(Y_test_bin)
    print('shape of Y_test_bin : ', Y_test_bin.shape)
    # -------
    
    # You can standarize the data : Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test) 


    from sklearn.neural_network import MLPClassifier

    # If your Y_train has multiple classes (Y_train_1D), it will use softmax to predict the output classes!
    # Else, it will use sigmoid at the last layer.

    # Default: L2 regularization, adam, 
    # If you want a one-layer hL0=1, hL1 =empty (Feedforward perceptron neural network  (FFNN) - (1 layer=shalow learning))
    hL0 = 5
    hL1 = 2
    # Problem : you don't know if it is doing He, or Xavier initialization. Also you do not not how the 
    # scalar random number you give for W is being used to randomly initialize W (a matrix the size of 
    #  the layers, not a scalar)
    w_int = np.random.permutation(hL0)[0]     # Random initialization
    model = MLPClassifier(solver='lbfgs', learning_rate_init=0.0075, hidden_layer_sizes=(hL0, hL1), random_state=w_int, max_iter=100)
    
    model.fit(X_train, Y_train_bin)

    Y_train_bin_predict = model.predict(X_train)
    Y_test_bin_predict = model.predict(X_test)
    #print('Y_test_bin_predict : ', Y_test_bin_predict)

    # The prediction probability of each class : is size [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train) # size is [n_samples, n_classes]
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    #Y_train_bin_pp = np.reshape(Y_train_bin_pp, (len(Y_train_1D_predict), len(unique_classes)))
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    #Y_test_bin_pp = np.reshape(Y_test_bin_pp, (len(Y_test_1D_predict), len(unique_classes)))
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp
    # OR
    # How confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)
    #Y_train_bin_score = model.decision_function(X_train)  # size is n_samples, 1
    #Y_test_bin_score = model.decision_function(X_test)

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_bin, Y_test_bin, Y_train_bin_predict, Y_test_bin_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score

# ----------------------------------------------

# Multilayer perceptron (MLP)/neural network (Deep Learning) : logistic regression NN
def multiclass_multilayer_perceptron_1Dinput(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    # You can standarize the data : Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test) 


    from sklearn.neural_network import MLPClassifier

    # If your Y_train has multiple classes (Y_train_1D), it will use softmax to predict the output classes!
    # Else, it will use sigmoid at the last layer.

    # Default: L2 regularization, adam, 
    # If you want a one-layer hL0=1, hL1 =empty (Feedforward perceptron neural network  (FFNN) - (1 layer=shalow learning))
    hL0 = 5
    hL1 = 2
    # Problem : you don't know if it is doing He, or Xavier initialization. Also you do not not how the 
    # scalar random number you give for W is being used to randomly initialize W (a matrix the size of 
    #  the layers, not a scalar)
    w_int = np.random.permutation(hL0)[0]     # Random initialization
    model = MLPClassifier(solver='lbfgs', learning_rate_init=0.0075, hidden_layer_sizes=(hL0, hL1), random_state=w_int, max_iter=100)
    
    model.fit(X_train, Y_train_1D)

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

# Multi-class Gaussian Naive Bayes
def multiclass_gaussian_naive_bayes(X_train, X_test, Y_train_1D, Y_test_1D):
    
    import numpy as np
    
    # X_train is size n_samples, n_features
    # Y_train is size [n_samples, 1]  where each class is a unique value
    
    from sklearn.naive_bayes import GaussianNB
    
    model = GaussianNB()
    
    model.fit(X_train, Y_train_1D)
    # OR
    # model.partial_fit(X_train, Y_train_bin, np.unique(Y))

    # -------
    # Y_predict : size [n_samples, 1]
    Y_train_1D_predict = model.predict(X_train)
    Y_test_1D_predict = model.predict(X_test)
    # -------
    
    # -------
    # The prediction probability of each class : size is [n_samples, n_classes]
    Y_train_bin_pp = model.predict_proba(X_train)
    Y_test_bin_pp = model.predict_proba(X_test)

    Y_train_bin_pp = np.array(Y_train_bin_pp)
    print('shape of Y_train_bin_pp : ', Y_train_bin_pp.shape)
    Y_test_bin_pp = np.array(Y_test_bin_pp)
    print('shape of Y_test_bin_pp : ', Y_test_bin_pp.shape)
    # -------
    
    # There is NO decision_function
    # ------------------------------
    Y_train_bin_score = Y_train_bin_pp
    Y_test_bin_score = Y_test_bin_pp

    Y_train_bin_score = np.array(Y_train_bin_score)
    print('shape of Y_train_bin_score : ', Y_train_bin_score.shape)
    Y_test_bin_score = np.array(Y_test_bin_score)
    print('shape of Y_test_bin_score : ', Y_test_bin_score.shape)
    
    return model, Y_train_1D_predict, Y_test_1D_predict, Y_train_bin_pp, Y_test_bin_pp, Y_train_bin_score, Y_test_bin_score
    
# ----------------------------------------------

 





# ----------------------------------------------
# Machine Learning Regression :
# ----------------------------------------------
# Polynomial Regression
def polynomial_regression(deg_of_f, X, Y):
    
    # Approximates coefficients for a linear function : 
    # x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
    P = np.polyfit(X, Y, deg_of_f)
    print('P : ', P)

    # Y_predict by hand
    if deg_of_f == 1:  # Linear model
        Y_predict = np.multiply(P[0], X) + P[1]
    elif deg_of_f == 2:   # 2nd order polynomial model
        XX = np.multiply(X, X)
        Y_predict = np.multiply(P[0], XX) + np.multiply(P[1], X) + P[2]
    
    # OR
    
    # Y_predict
    Y_predict = np.poly1d(P)

    return model, Y_test_predict

# ----------------------------------------------

# Linear Regression : ordinary difference model
def linear_regression(X_train, X_test, Y_train_1D):
    # X_train is size (n_samples, n_features)
    # X_test is size (n_samples, n_features)
    # Y_train_1D is size (n_samples, 1) where the n_classes are represented by 1,2,3,..., etc

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, Y_train_1D)

    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    # R-squared
    train_rsquared = R_squared(Y_train_1D, Y_train_predict)
    print('The R-squared training value : ' + str(train_rsquared))

    return model, Y_test_predict

# ----------------------------------------------

# LogisticRegression binary
def logistic_regression(X_train, X_test, Y_train_1D):
    # X_train is size (n_samples, n_features)
    # X_test is size (n_samples, n_features)
    # Y_train_1D is size (n_samples, 1) where the n_classes are represented by 1,2,3,..., etc

    from sklearn.linear_model import LogisticRegression

    #For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
    #For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
    #‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
    #‘liblinear’ and ‘saga’ also handle L1 penalty
    #‘saga’ also supports ‘elasticnet’ penalty
    solver = 'lbfgs' # default:'lbfgs' # 'newton-cg' 'saga'
    model_type = 'ovr'   # default:'ovr' , 'multinomial'
    
    # Set regularization rate
    reg = 0.01
    model = LogisticRegression(C=1/reg, solver="liblinear")
    # model = LogisticRegression(C=1, solver=solver, multi_class=model_type, random_state=0)  #  n_jobs=-1
    # model = LogisticRegression(class_weight='balanced', max_iter=1000)  # balance imbalanced clases

    model.fit(X_train, Y_train_1D)
    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    # R-squared
    #train_rsquared = accuracy_score(Y_train_1D, Y_train_predict)  #from blog post
    #OR
    train_rsquared = score(X_train, Y_train_1D)  # it does fit and predict, then calculates R-squared
    #OR
    #train_rsquared = R_squared(Y_train_1D, Y_train_predict)
    print('The R-squared training value : ' + str(train_rsquared))

    return model, Y_test_predict

# ----------------------------------------------

# Multiple Output Regression - linear/logistic regression for multiple outputs
def multi_output_regression(X_train, Y_train, X_test):
    
    from sklearn.datasets import make_regression
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    model.fit(X_train, Y_train)

    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)
    
    return model, Y_test_predict

# ----------------------------------------------

def model_XGBRegressor(X_train, Y_train_1D, X_test, Y_test_1D):
    from xgboost import XGBRegressor
    

    model = XGBClassifier()

    model.fit(X_train, Y_train_1D)

    Y_test_predict = model.predict(X_test)
    
    f1_test = metrics.f1_score(Y_test_1D, Y_test_predict, average='micro')
    print('f1_test : ', f1_test)
    
    return model, Y_test_predict
    
# ----------------------------------------------

# Decision Tree Regressor
def Decision_Tree_Regressor(X_train, X_test, Y_train_1D, Y_test_1D):
    from sklearn.tree import DecisionTreeRegressor
    
    
    
    # Hyperparameters
    
    # 0. Default
    # model = DecisionTreeRegressor()
    # model = DecisionTreeRegressor(random_state=1)  # ensures you get the same results in each run
    
    # 1. Maximum tree depth : number of levels of splitting
    model = DecisionTreeRegressor(max_depth=4) # minimization de 'overfitting', 
    
    # 2. Minimum node size: min nombre des points des données
    model = DecisionTreeRegressor(max_depth=6, min_samples_split=3)
    
    # 3. Minimum leaf size
    model = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=3)
    
    # 4. Maximum features: maximum number of features to consider at each split
    model = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=3, max_features=3)
    
    
    model = model.fit(X_train, Y_train_1D)
    
    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)
    
    # R-squared : comparison of y with y_prediction
    n_classes = Y_train_1D.shape[1]
    out = []
    for i in range(0,n_classes):
        out.append(R_squared(y, yhat))
    print("train accuracy (R-squared): " + str(out) + "%")
    
    
    # ...............................
    #  Create a visualization of the decision tree
    # Remember that the number of features used (feature from the far right to the far left of the X matrix) depends on the max_depth. (ie: if the X matrix has 5 features ['a', 'b','c','d','e'] but the max_depth=3 it will use features ['c','d','e'] to construct the model)
    feature_names = X_train.columns
    print('feature_names: ', feature_names)
    class_names = Y_train_1D.columns
    print('class_names: ', class_names)
    
    from sklearn import tree
    text_representation = tree.export_text(model, feature_names=feature_names)
    print(text_representation)

    from sklearn import tree
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize = (8, 4), dpi=400)
    _ = tree.plot_tree(
      model,
      feature_names = feature_names,
      class_names = class_names,
      filled = True
    )
    # ...............................
    
    return model, Y_test_predict



# ----------------------------------------------
# Example
from sklearn.preprocessing import LabelEncoder
X = ht_metrics_pd_df.drop(['device_id'], axis=1).drop(['avg_vo2'], axis=1)
X['lifestyle_num'] = LabelEncoder().fit_transform(X['lifestyle'])
X = X.drop(['lifestyle'], axis=1)
y = ht_metrics_pd_df['avg_vo2']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 

dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train, Y_train)

# ----------------------------------
laquelle = 0
if laquelle == 0:
  from sklearn.metrics import r2_score
  y_train_predicted = dt.predict(X_train)
  y_test_predicted = dt.predict(X_test)
  rsquared_train = r2_score(Y_train, y_train_predicted)
  rsquared_test = r2_score(Y_test, y_test_predicted)
elif laquelle == 1:
  rsquared_train = dt.score(X_train, Y_train)
  rsquared_test = dt.score(X_test, Y_test)
# ----------------------------------

print("R2 on training set: ", round(rsquared_train, 3) )
print("R2 on test set: ", round(rsquared_test, 3) )
# ----------------------------------------------




# ----------------------------------------------

def model_XGBRegressor(X_train, Y_train, X_test, Y_test):
    from xgboost import XGBRegressor
    
    # n_estimators (specifies how many times to go through the modeling cycle described above) : value too low = underfitting, value too high = overfitting
    
    # learning_rate (is the rate of gradient descent) : typical range of values are 100-1000
    
    # early_stopping_rounds (offers a way to automatically find the ideal value for n_estimators - if you want to stop doing descent after the loss remains constant)
    
    # ie : early_stopping_rounds=5 means gradient descent stops after 5 straight rounds of deteriorating validation scores
    
    # When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter
    
    # If you have a smaller learning_rate (ie: 0.1) you take smaller steps to minimize the loss function, so you need to interate gradient descent longer (thus making n_estimators=a higher value, 1000)
    
    # n_jobs (the number of cores on your machine such that you can parallelise the model training)
    
    # https://xgboost.readthedocs.io/en/latest/
    
    # Default
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    # Setting a maximum iteration
    # model = XGBRegressor(n_estimators=500)
    # model.fit(X_train, Y_train)
    # OR
    # Early stopping : gradient descent stops after 5 straight rounds of deteriorating validation scores
    # model = XGBRegressor(n_estimators=500)
    # model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=[(X_test, Y_test)], verbose=False)
    
    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error
    print("Mean Absolute Error: " + str(mean_absolute_error(Y_test_predict, Y_test)))
    
    return model, Y_test_predict


# ----------------------------------------------
    
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(criterion="mae", random_state=0, class_weight='balanced') 
# OU
# Defines whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
# Motivation pour utiliser: il y a moins "d'overfitting"
model = RandomForestRegressor(n_estimators=50, max_depth=8, bootstrap=True)
# OU
# No bootstrap samples,performance gets slightly worse.
model = RandomForestRegressor(n_estimators=50, max_depth=8, bootstrap=False)

model.fit(train_X, train_y)
pred = model.predict(val_X)


# ----------------------------------------------




# ----------------------------------------------
# Machine Learning Explainability :
# ----------------------------------------------
def pipeline_permutation_importance(model, X_test, Y_test, feature_names):
    
    # Benefits of permutation importance : 1) fast to calculate, 2) widely used and understood, 3) consistent with properties we would want a feature importance measure to have.

    # Permutation importance is calculated after a model has been fitted:
    # 1) Get a trained model
    # 2) Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
    # 3) Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column
    
    from sklearn.inspection import permutation_importance
    
    # How to do we know which features are important? 
    # 1) test different configuration of X and see which gives the best prediction (what I did before)
    # 2) permutation method (permute the rows of each feature to see if it changes the prediction value)

    # ----------------
    # Permutation importance of features : probe which features are most predictive
    # ----------------
    # The permutation feature importance is the decrease in a model score 
    # when a single feature value is randomly shuffled. 

    # it is the difference between the mean accuracy using all the features and the mean 
    # accuracy of each feature shuffled

    # The difference that is positive and largest, means that the feature is important 
    # because without the feature in proper order the model can not predict well on the 
    # validation data.

    # Example for feature_names :
    # X_test.columns.tolist()
    # feature_names = ['joy', 'joy1derv', 'joy2derv', 'fres', 'freq_t', 'freq_fres']

    r = permutation_importance(model, X_test, Y_test, n_repeats=10, random_state=0, scoring='accuracy')

    ovtot = []
    for i in r.importances_mean.argsort()[::-1]:
        outvals = feature_names[i], r.importances_mean[i], r.importances_std[i]
        ovtot.append(outvals)

    print('ovtot : ', ovtot)

    return ovtot
# ----------------------------------------------

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(X_test, Y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

# The values with the largest weight are the most important features, the weight is the amount that the model performance changed with shuffling (positive weight = decreased in accuracy, negative weight = increased in accuracy, meaning not important OR chance predictions)

# ----------------------------------------------


# ----------------------------------------------
# feature importance 0 : determining which features have the biggest impact on predictions


# ----------------------------------------------





# ----------------------------------------------
# Feature imortance 1: Importance des caractéristiques utilisant les moindres carrés ordinaires

# if X and y are pandas DataFrames, be sure to reset index
X.reset_index(drop=True,inplace=True)
y.reset_index(drop=True,inplace=True)

import statsmodels.api as sm
model = sm.OLS(endog=y, exog=X)
bmi_ols_results = model.fit()

bmi_ols_results.summary()  # Voir toutes des results statistiques 


# Organizer des features from smallest to largest p-value
# Organizer des features de la plus petite à la plus grande p-valeur 
df_res = pd.DataFrame([list(bmi_ols_results.params.index), list(bmi_ols_results.params)]).T
df_res.columns = ['feat', 'coef']
d_pvals = dict(zip(list(bmi_ols_results.pvalues.index), list(bmi_ols_results.pvalues)))

df_res['p_vals'] = [d_pvals[df_res.feat.iloc[i]] for i in range(len(df_res))]

# Le p-valeur plus petite signifie que le feature a un valeur significantment different que zero. Si le valeur de feature est different que zero, le feature influencera le modele. Donc, des features avec un p-valeur plus petite sont plus important pour la prediction de modele. 
df_res.sort_values(by='p_vals', ascending=False)

# ----------------------------------------------


# ----------------------------------------------
# Feature imortance 2: Regularization

from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=10)
lasso_reg.fit(X, y)

lasso_reg.score(X, y)

import pandas as pd
# Lister des coefficients de coefficient plus grand à plus petite
pd.DataFrame(list(zip(lasso_reg.coef_, X.columns)), columns=['coefficient', 'feature_name']).sort_values('coefficient', ascending=False)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# Lister des coefficients de coefficient plus grand à plus petite
pd.DataFrame(list(zip(lr.coef_, X.columns)), columns=['coefficient', 'feature_name']).sort_values('coefficient', ascending=False)

# Apres vous ordrez des coefficients, refaites le modele avec des characteristiques plus grand pour verifier la vrai contribution de characteristique
X = train_df[['dummy_Athlete']]
y = train_df['avg_bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
lr = Lasso(alpha=.01)
lr.fit(X_test, y_test)
lr.score(X_test, y_test)

# ----------------------------------------------


# ----------------------------------------------
# Feature relationship: corr
df_temp['col0'].corr(df_temp['col1'], method='pearson')

# ----------------------------------------------



# ----------------------------------------------

# Partial dependence plots : show how a feature affects predictions
    
# partial dependence plots are calculated after a model has been fit.  use the fitted model to predict our outcome, then we repeatedly alter the value for one variable to make a series of predictions.

# The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.
# A blue shaded area indicates level of confidence

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# 1D partial dependence plot: Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_test, model_features=feature_names, feature='col_feature_nom')
pdp.pdp_plot(pdp_goals, 'col_feature_nom')
plt.show()

# 2D partial dependence plots : the interactions between two features
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

# ----------------------------------------------

# PDP is showing you how your model changes your features.  You could have noise for features and your model could tell you that you have something good, perhaps 
    
# To see how the pdp function operates : 

# The rate of partial dependence plot features does not necessarily meant it is more important, it indicates variance in the prediction

# On the partial dependence plot, x-axis=one of the features (X1), y-axis=y
# If you want to control the dependence (have a good feature that influences y) - you change y with respect to X (the feature) 

# For example: create a y so that our PDP plot has a positive slope in the range [-1,1], and a negative slope everywhere else.


# The 2 influences is the slope of the points before  X1=-1 and after X1=1 - it gives the negative slope
n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2
# Create y. you should have X1 and X2 in the expression for y
y = np.ones(n_samples)  # The permutation dependence plot shows zero 
y = -2 * X1 * (X1<-1) + X1 - 2 * X1 * (X1>1) - X2  # The pdp shows what you want

# create dataframe because pdp_isolate expects a dataFrame as an argument
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = my_df.drop(['y'], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)

pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')

# visualize your results
pdp.pdp_plot(pdp_dist, 'X1')
plt.show()

# ----------------------------------------------

# The point of this exercise : you can have a model that says that a feature is important (based on permuting the rows) but there is no relationship between the feature and model prediction (based on the pdp) 
 
# This happens when your features are random noise () OR you have features that are almost identical but inverted (ie : a feature where something increases and another feature where something decreases almost the same)

# In this case, remove noise like features and reduntant features -- then  redo the model and permutation/pdp
# Check the distribution : noise features have a distribution of zero, inverted similar features have a inverted gaussian distribution

# Exercise : Create a dataset with 2 features and a target, such that the pdp of the first feature is flat, but its permutation importance is high. We will use a RandomForest for the model.

# If you want pdp of X1 to be flat, but have high weight

# y = 0 for some combination of X1 and X2
# weight = when you change the row order of X1 it changes y drastically 

X1 = [(i/n_samples)+np.sin(0.001*i)*1.00000001 for i in range(n_samples)]
print('length of X1 :', len(X1))
X2 = [-(i/n_samples+1)+np.sin(-0.001*i) for i in range(n_samples)]
print('length of X2 :', len(X2))
# Create y. you should have X1 and X2 in the expression for y
y_canceling_sin = [X1[i]+X2[i] for i in range(n_samples)]

# OR

y0 = [X1[i]*X1[i] for i in range(n_samples)]
# subtract off slope
out = np.polyfit(list(range(len(y0))), y0, 1)
slo = [out[0]*i + 0 for i in range(n_samples)]
ysin = [y0[i] - slo[i] for i in range(n_samples)]

# OR

from scipy import signal
X11= 4 * rand(n_samples) - 2
n = 4   # filter order
fs = 250 # data sampling frequency (Hz)
fc = 0.1  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(n, w, 'low')  # 3rd order
filtsig = 4 * signal.filtfilt(b, a, X11)
yrand = [filtsig[i]*filtsig[i] for i in range(n_samples)]

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=2, cols=1)
config = dict({'scrollZoom': True, 'displayModeBar': True, 'editable': True})

xxORG = list(range(len(X1)))
fig.add_trace(go.Scatter(x=xxORG, y=X1, name='X1', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=xxORG, y=X2, name='X2', line = dict(color='green', width=2, dash='solid'), showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=xxORG, y=filtsig, name='filtsig', line = dict(color='blue', width=2, dash='solid'), showlegend=True), row=1, col=1)

fig.add_trace(go.Scatter(x=xxORG, y=ysin, name='y', line = dict(color='red', width=2, dash='solid'), showlegend=True), row=2, col=1)
fig.add_trace(go.Scatter(x=xxORG, y=y_canceling_sin, name='yidea', line = dict(color='green', width=2, dash='solid'), showlegend=True), row=2, col=1)
fig.add_trace(go.Scatter(x=xxORG, y=yrand, name='yrand', line = dict(color='blue', width=2, dash='solid'), showlegend=True), row=2, col=1)

fig.update_layout(title='', xaxis_title='data points', yaxis_title='')
fig.show(config=config)

import seaborn as sns
import matplotlib.pyplot as plt
fig, ax=plt.subplots(3,1)
sns.distplot(ysin, kde=False, label="ysin", color="r", ax=ax[0])
sns.distplot(y_canceling_sin, kde=False, label="y_canceling_sin", color="b", ax=ax[1])
sns.distplot(yrand, kde=False, label="yrand", color="g", ax=ax[2])

# ----------------------------------------------

# SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature.
    
# SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value 
    
# Permutation transforms the feature into a baseline value, but it does not compare how the feature is different from the baseline balue. 
    
# SHAP tells how the orginal feature is different from the baseline value
    
# sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values

# ----------------------------------------------



# ----------------------------------------------

def sort_dict_by_value(d, reverse = False):
    return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))


import itertools
import math
import numpy as np


def SHAP_byhand_feature_importance(X_test, model, feat_nom, X_train, y_train):
    
    num_of_feat = X_test.shape[1]
    print('number of features: ', num_of_feat)

    y_levels = []
    y_levels_labels = []

    # zeroth level
    y = model.predict(X_test)
    y_levels.append(y)
    y_levels_labels.append([999])

    num_of_levels = num_of_feat+1

    # Get predictions for all levels : where each level is a combination of numbered features 
    # (ordering does not matter)
    for r in range(1, num_of_levels):
        vec = np.arange(num_of_feat)
        strinput = [str(i) for i in vec]
        combs = list(itertools.combinations(strinput, r))
        combs_list = [[int(k) for k in combs[j]] for j in range(len(combs))]

        temp_lev = []
        for j in range(len(combs)):
            # now we need to use j tuple as the index for grabing features

            # convert string tuple to a numbered list
            ind = [int(i) for i in combs[j]]
            # Get new feature matrix with combinatorial features
            X_test_temp = X_test[:,ind]

            which_way = 1
            if which_way == 0:
                # Way 0: use same model
                # pad with the same featues, so the model can predict with the same number of features
                feat_manque = num_of_feat - len(ind)
                rr = np.random.choice(len(ind), feat_manque)
                extra = [ind[i] for i in rr]
                X_test_temp2 = np.concatenate((X_test_temp, X_test[:,extra]), axis=1)
                predictions = model.predict(X_test_temp2)

            elif which_way == 1:
                # Way 1: use NEW model - what I did
                txt = str(type(model))
                
                if txt.find('LinearRegression') != -1:
                    model = LinearRegression()
                elif txt.find('XGBClassifier') != -1:
                    model = XGBClassifier()
                else:
                    model = XGBClassifier()
                X_train_temp = X_train[:,ind]
                model.fit(X_train_temp, y_train)
                predictions = model.predict(X_test_temp)


            # Could do the subtraction here
            temp_lev.append(predictions)

        y_levels.append(temp_lev)
        y_levels_labels.append(combs_list)
    
    print('Combinations of features levels : ', y_levels_labels)
    
    # --------------------------------

    # Now, get Margional contributions
    n = num_of_feat

    # This one is always subtracted for each feature
    MC_1st = [np.sum(y_levels[1][i] - y_levels[0]) for i in range(num_of_feat)]
    r = len(y_levels_labels[0])
    w = (r * math.comb(n, r))**(-1)

    SHAP = []
    for f in range(num_of_feat):

        # say, we are on feature 0
        base_feat = y_levels_labels[1][f][0]

        MC_feat_level = []
        # need to cycle over each level using a each feature!
        tot = 0
        for i in range(2, len(y_levels)):  # 2-6 

            cur = y_levels_labels[i]
            r = len(y_levels_labels[i][0])
            w = (r * math.comb(n, r))**(-1)

            MC_feat = []

            for ind, i2 in enumerate(cur):  # i get each nested list [0, 1], so i2 = [0, 1]
                if base_feat in i2:
                    MC_feat.append(w*np.sum(y_levels[i][ind] - y_levels[1][base_feat] ))  
                    tot = tot + 1
                    
            MC_feat_level.append(np.sum(MC_feat))  # sum of MC per level
        
        # sum SHAP value per feature
        # SHAP.append(w*MC_1st[f] + np.sum(MC_feat_level))
        
        # OR
        
        # mean SHAP value per feature: sum all level-sums together and take the mean
        SHAP.append((w*MC_1st[f] + np.sum(MC_feat_level))/(tot+1))

    # A small SHAP feature value means that it contribues to the overall prediction
    vals = dict(zip(feat_nom, SHAP))

    # -------------------------

    # Sort the columns from smallest to largest normalized SHAP value
    marquers_important = sort_dict_by_value(vals, reverse = True)

    # -------------------------
    
    return marquers_important
    
# ----------------------------------------------












# ----------------------------------------------
# Regression Evaluation :
# ----------------------------------------------
def R_squared(y, yhat):
    SSres = np.square(np.sum(y - np.mean(y)))
    SStot = np.square(np.sum(y - yhat))
    
    R_squared = 1 - SSres/SStot
    
    return R_squared

train_rsquared = R_squared(Y_train_1D, Y_train_predict)
print('The R-squared training value : ' + str(train_rsquared))
# -------------------------

from sklearn.metrics import r2_score
rsquared = r2_score(y_true, y_pred)

# -------------------------

rsquared = model.score(X_train, Y_train_1D)  # it does fit and predict, then calculates R-squared

# -------------------------




# ----------------------------------------------


# Classification evaluation : 

# ----------------------------------------------

def handmade_accuracy(X, Y_1D, Y_1D_predict):
    # 0) Accuracy percentage : classification
    cor = 0
    n_samples = X.shape[0]
    for i in range(0,n_samples):
        if np.sum(Y_1D_predict[i] - Y_1D[i]) == 0:
            cor = cor + 1
    
    Accuracy = (cor/n_samples)*100
    print("Accuracy: " + str(Accuracy) + "%")
    
    return Accuracy

# ----------------------------------------------


# ----------------------------------------------

from sklearn.metrics import accuracy_score
train_rsquared = accuracy_score(Y_train_1D, Y_train_predict, normalize=True)

# ----------------------------------------------


# ----------------------------------------------

# Assign someone to class 0 if the model says the probability of belonging to that class is greater than 70%
def adjust_y_pred(y_predicted_proba, percentage_thresh):
    # percentage_thresh = 0.7
    y_pred_adj = [0 if x > percentage_thresh else 1 for x in y_predicted_proba]
    
    return y_pred_adj

# ----------------------------------------------





# ----------------------------------------------
# Class balancing
# ----------------------------------------------


# ------------------------------
# OVERSAMPLING
# ------------------------------
!pip install imbalanced-learn
# OU
%pip install imbalanced-learn # pour Databricks

# set the sampling strategy to minority which will make the minority class the same size as the majority class.
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
y_over.value_counts()
# ------------------------------


# ------------------------------
def pad_data_2makeclasses_equivalent(df_feat):
    # Remove nan value per row
    # df_test_noNan = df_feat.dropna(axis=0)
    # OR
    df_test_noNan = my_dropna_python(df_feat)
    
    del df_feat
	
    # ----------------

    # Confirm that there are no nan values
    out = df_test_noNan.isnull().values.any()
    print('Are there nan valeus in the data : ', out)

    # ----------------

    # Check class balance
    needed_samps_class, counted_value, count_index, st, endd = count_classes(df_test_noNan)

    # ----------------

    print('shape of dataframe before padding : ', df_test_noNan.shape)

    # ----------------
    # Pad the DataFrame
    timesteps = len(df_test_noNan.num.unique())
    print('timesteps: ', timesteps)
    n_classes = len(count_index)
    
    df_2add_on = pd.DataFrame()

    for i in range(n_classes):
        # Pad short length classes
        for j in range(needed_samps_class[i]):
            # print('j: ', j)
            flag = 0
            
            while flag == 0:
                choix = df_test_noNan.index[(df_test_noNan.num == 0) & (df_test_noNan.y == i)].to_numpy()
                lq = np.random.permutation(len(choix))[0]
                # print('lq: ', lq)
                
                # Append the data with padded data entry
                df_coupe = df_test_noNan.iloc[choix[lq]:choix[lq]+timesteps, :]

                # To prevent repeating the same sample often :  
                if j == 0:
                    index_prev = choix[lq]
                    df_2add_on = pd.concat([df_2add_on, df_coupe], axis=0)
                    flag = 1 # to brake while
                else:
                    if len(choix) == 1:
                        # Do not obtain a shuffled assortment of data
                        index_prev = choix[lq]
                        df_2add_on = pd.concat([df_2add_on, df_coupe], axis=0)
                        flag = 1 # to brake while
                    else:
                        # Obtain a shuffled assortment of data
                        if choix[lq] != index_prev:
                            index_prev = choix[lq]
                            df_2add_on = pd.concat([df_2add_on, df_coupe], axis=0)
                            flag = 1 # to brake while
				
    # ----------------

    # DataFrame a besoin les noms de columns d'avoir le meme noms que df_test_noNan
    df_2add_on = df_2add_on.reset_index(drop=True)  # reset index : delete the old index column

    df_2add_on.columns = df_test_noNan.columns

    print('shape of dataframe to add to original dataframe: ', df_2add_on.shape)

    # ----------------

    # want to arrange the dataframe with respect to rows (stack on top of the other): so axis=0 
    # OR think of it as the rows of the df change so you put axis=0 for rows
    df_test2 = pd.concat([df_test_noNan, df_2add_on], axis=0)
    df_test2 = df_test2.reset_index(drop=True)  # reset index : delete the old index column

    print('shape of padded dataframe (original + toadd) : ', df_test2.shape)

    del df_test_noNan, df_2add_on
    gc.collect()

    # ----------------

    # Final check of class balance
    needed_samps_class, counted_value, count_index, st, endd = count_classes(df_test2)
    
    # ----------------
    
    return df_test2

# ------------------------------

def count_classes(df_test_noNan):

    # Get start and end index values for each sample
    num = list(map(int, df_test_noNan.num.to_numpy()))
    st = df_test_noNan.index[(df_test_noNan.num == 0)].to_numpy()
    endd = df_test_noNan.index[(df_test_noNan.num == df_test_noNan.num.max())].to_numpy()
    
    # ----------------

    yy = list(map(int, df_test_noNan.y.to_numpy()))
    y_short = []
    for i in range(len(st)):
        y_short.append(yy[st[i]:st[i]+1][0])

    # ----------------

    liste = Counter(y_short).most_common()
    count_index, counted_value = list(map(list, zip(*liste)))
    print('Before sorting counted_value : ', counted_value)
    print('Before sorting count_index : ', count_index)

    # ----------------

    # Sort counted_value by count_index; in ascending order
    sind = np.argsort(count_index)
    count_index = [count_index[i] for i in sind]
    counted_value = [counted_value[i] for i in sind]
    print('After sorting counted_value : ', counted_value)
    print('After sorting count_index : ', count_index)

    # ----------------

    # Determine how much to pad each class label
    needed_samps_class = np.max(counted_value) - counted_value
    print('needed_samps_class : ', needed_samps_class)

    # ----------------
    
    return needed_samps_class, counted_value, count_index, st, endd

# ------------------------------

def my_dropna_python(df):
    # Python
    col_names = list(df.columns.values)
    # OR
    # col_names = list(df.columns)
    
    df = df.to_numpy()
    df = np.array(df, dtype=object)
    # print('size of df : ', df.shape)
    data = []
    num_of_cols = df.shape[1]
    for i in range(df.shape[0]):
        row_vec = df[i,:]
        
        out = [isnan(row_vec[i]) for i in range(len(row_vec))]
        # OR
        # out = []
        # for i in range(len(row_vec)):
            # print('row_vec[i]', row_vec[i])
            # out.append(isnan(row_vec[i]))
        
        out = make_a_properlist(out)  # for dataframes with nested arrays
        
        if any(out) == False:
            data.append(df[i,:])
    
    num_of_rows = len(data)
    data0 = np.reshape(data, (num_of_rows, num_of_cols))
    
    df_new = pd.DataFrame(data=data0, columns=col_names)
    
    return df_new

# ------------------------------


# ------------------------------
# UNDERSAMPLING
# ------------------------------
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy='majority') # on veut la taille de la classe majoritie egale à la taille de la classe minoritie
X_under, y_under = undersample.fit_resample(X, y)
y_under.value_counts()

# we want the majority class to be a ratio of as the minority
# On veut que le class major est un ratio de la taille de la classe minoritie
undersample_2 = RandomUnderSampler(sampling_strategy=0.75)
X_under2, y_under2 = undersample_2.fit_resample(X, y)
y_under2.value_counts()


# ------------------------------
# MODEL WEIGHTS
# ------------------------------
# Sklearn has a built in utility function that will calculate weights based on class frequencies. It does this by automatically weighting classes inversely proportional to how frequently they appear in the data.


# Façon 0: Default (None):
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight=None)


# Façon 1: Balanced
rf = RandomForestClassifier(class_weight="balanced")


# Façon 2: Balanced subsample
rf = RandomForestClassifier(class_weight="balanced_subsample")


# Façon 3: Dictionary of ratios:
# We can calculate the exact ratio we would use to evenly balance the classes, and use that in our class weight dictionary. We can use the sklearn.utils class_weight function to accomplish this.
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1], y=y)
print(weights)

class_weights_dict = dict(enumerate(weights))
print(class_weights_dict)

rf = RandomForestClassifier(class_weight=class_weights_dict)
# OU
rf = RandomForestClassifier(class_weight={0: 999, 1: 0.0009})


rf.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, rf.predict(X_test)))






# ----------------------------------------------
# Evaluation Methods
# ----------------------------------------------

def evaluation_methods_multi_class_bin(model, X, Y_bin, Y_bin_predict, Y_bin_pp, Y_bin_score):
    
    # 1) cross_val_score with scoring : 
    from sklearn.model_selection import cross_val_score
    cv_num = 5
    acc_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="accuracy")
    # print('acc_crossval : ', acc_crossval)

    prec_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="precision")
    # print('prec_crossval : ', prec_crossval)

    recall_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="recall")
    # print('recall_crossval : ', recall_crossval)

    # Multiclass case :
    rocauc_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="roc_auc_ovo")
    # print('rocauc_crossval : ', rocauc_crossval)

    roc_auc_ovo_weighted_crossval = cross_val_score(model, X, Y_bin, cv=cv_num, scoring="roc_auc_ovo_weighted")
    # print('roc_auc_ovo_weighted_crossval : ', roc_auc_ovo_weighted_crossval)
    
    
    # ----------------------------
    
    # Collapse the Y_bin into a Y_1D
    Y_1D = debinarize_Ybin_2_Y1Dvec(Y_bin)
    # print('shape of Y_1D : ', Y_1D.shape)
    
    # Collapse the Y_bin_predict into a Y_1D_predict
    Y_1D_predict = debinarize_Ybin_2_Y1Dvec(Y_bin_predict)
    # print('shape of Y_1D_predict : ', Y_1D_predict.shape)
    
    import numpy as np
    
    # Ensure vector is of integers
    Y_1D = [int(i) for i in Y_1D]
    Y_1D_predict = [int(i) for i in Y_1D_predict]
    
    # Number of samples
    m_examples = len(Y_1D)

    # Number of classes
    temp = np.unique(Y_1D)
    unique_classes = [int(i) for i in temp]
    
    # ----------------------------
    
    # 2) Confusion matrix
    from sklearn.metrics import confusion_matrix
    matrix_of_counts = confusion_matrix(Y_1D, Y_1D_predict)
    # print('matrix_of_counts : ', matrix_of_counts)
    # OR
    matrix_normalized = confusion_matrix(Y_1D, Y_1D_predict, normalize='all')
    # print('matrix_normalized : ', matrix_normalized)

    # Display the confusion matrix
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # disp = ConfusionMatrixDisplay(confusion_matrix=matrix_of_counts, display_labels=clf.classes_)
    # disp.plot()
    # plt.show()
    
    # ----------------------------
    
    # 3) Classification report : builds a text report showing the main classification metrics
    from sklearn. metrics import classification_report
    print(classification_report(Y_1D, Y_1D_predict))
    
    # precision    recall  f1-score   support
    #        0       0.81      0.88      0.85      2986
    #        1       0.72      0.60      0.66      1514
    # 
    # accuracy                           0.79      4500
    # macro avg       0.77      0.74      0.75      4500
    # weighted avg       0.78      0.79      0.78      4500
    
    # ----------------------------

    # 4) Direct calculation of metrics
    from sklearn import metrics

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    # labels = array-like, default=None

    # pos_label = str or int, default=1 The class to report if average='binary' and the data is binary. 
    # If the data are multiclass or multilabel, this will be ignored

    # average = ['binary', 'micro', 'macro', 'weighted', 'samples', None]
    # This parameter is required for multiclass/multilabel targets. 
    # None : the scores for each class are returned
    # 'binary'  : Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    # 'micro' : Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro' : Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted' : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    # 'samples' : Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    average = 'micro'

    # sample_weight : array-like of shape (n_samples,), default=None
    acc_dircalc = metrics.accuracy_score(Y_1D, Y_1D_predict)
    # print('acc_dircalc : ', acc_dircalc)

    prec_dircalc = metrics.precision_score(Y_1D, Y_1D_predict, average=average)
    # print('prec_dircalc : ', prec_dircalc)

    recall_dircalc = metrics.recall_score(Y_1D, Y_1D_predict, average=average)
    # print('recall_dircalc : ', recall_dircalc)

    f1_dircalc = metrics.f1_score(Y_1D, Y_1D_predict, average=average)
    # print('f1_dircalc : ', f1_dircalc)

    # beta=0.5, 1, 2
    fbeta_dircalc = metrics.fbeta_score(Y_1D, Y_1D_predict, beta=0.5, average=average)
    # print('fbeta_dircalc : ', fbeta_dircalc)

    # y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
    # y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
    rocauc_pp_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_pp, average=average) # prediction probability
    # print('rocauc_pp_dircalc : ', rocauc_pp_dircalc)
    # OR
    rocauc_df_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_score, average=average) # decision function 
    # print('rocauc_df_dircalc : ', rocauc_df_dircalc)

    # ----------------------------
    
    # 5) Direct calculation of metrics : micro-average ROC curve and ROC area
    # True Positive Rate (TPR) = TP / (TP + FN) = efficiency (εₛ) to identify the signal (also known as Recall or Sensitivity)
    
    # False Positive Rate (FPR) = FP / (FP + TN) = inefficiency (ε_B) to reject background
    
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    from sklearn.metrics import roc_curve, auc
    
    laquelle = 1
    
    if laquelle == 0:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(unique_classes)):
            fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_score[:, i]) # decision function
            # OR
            # fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_pp[:, i]) # prediction probability
            
            roc_auc[i] = auc(fpr[i], tpr[i])
    elif laquelle == 1:
        # Micro-average ROC curve and ROC area : all the classes together!
        fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_score.ravel()) # decision function
        # OR
        # fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_pp.ravel()) # prediction probability
        
        roc_auc0 = auc(fpr0, tpr0)
    
    
    plotOUpas = 0
    if plotOUpas == 1:
        # Plot of a ROC curve for a specific class
        import matplotlib.pyplot as plt
        plt.figure()
        lw = 2
        plt.plot(fpr0, tpr0, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc0)
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()
    
    value_pack = {}
    
    var_list = ['acc_crossval', 'prec_crossval', 'recall_crossval', 'roc_auc_ovo_weighted_crossval', 'acc_dircalc', 'prec_dircalc', 'recall_dircalc', 'f1_dircalc', 'fbeta_dircalc', 'rocauc_pp_dircalc', 'rocauc_df_dircalc']
    var_list_num = [acc_crossval, prec_crossval, recall_crossval, roc_auc_ovo_weighted_crossval, acc_dircalc, prec_dircalc, recall_dircalc, f1_dircalc, fbeta_dircalc, rocauc_pp_dircalc, rocauc_df_dircalc]
    
    for q in range(len(var_list)):
        value_pack['%s' % (var_list[q])] = var_list_num[q]
    
    return value_pack
    
# ----------------------------------------------

def evaluation_methods_multi_class_1D(model, X, Y_1D, Y_1D_predict, Y_bin_pp, Y_bin_score):
    
    import numpy as np
    
    # -------
    # Need to binarize Y into size [n_samples, n_classes]
    Y_bin, unique_classes = binarize_Y1Dvec_2_Ybin(Y_1D)
    Y_bin_predict, unique_classes = binarize_Y1Dvec_2_Ybin(Y_1D_predict)

    Y_bin = np.array(Y_bin)
    print('shape of Y_bin : ', Y_bin.shape)

    Y_bin_predict = np.array(Y_bin_predict)
    print('shape of Y_bin_predict : ', Y_bin_predict.shape)
    # -------
    
    # 1) cross_val_score with scoring : 
    from sklearn.model_selection import cross_val_score
    cv_num = 5
    acc_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="accuracy")
    # print('acc_crossval : ', acc_crossval)
    
    prec_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="precision")
    # print('prec_crossval : ', prec_crossval)

    recall_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="recall")
    # print('recall_crossval : ', recall_crossval)
    
    # Multiclass case :  ** check if Y_bin works
    rocauc_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="roc_auc_ovo")
    # print('rocauc_crossval : ', rocauc_crossval)

    roc_auc_ovo_weighted_crossval = cross_val_score(model, X, Y_1D, cv=cv_num, scoring="roc_auc_ovo_weighted")
    # print('roc_auc_ovo_weighted_crossval : ', roc_auc_ovo_weighted_crossval)
    
    # ----------------------------
    
    # Model performance
    baseline_model_cv = cross_validate(model, X, Y_1D, cv=StratifiedKFold(n_splits=5), n_jobs=-1, scoring="recall")
    print(f"{baseline_model_cv['test_score'].mean():.3f} +/- {baseline_model_cv['test_score'].std():.3f}")
    
    # ----------------------------

    # Number of classes
    temp = np.unique(Y_1D)
    unique_classes = [int(i) for i in temp]
    
    # ----------------------------
    
    # 2) Confusion matrix
    from sklearn.metrics import confusion_matrix
    matrix_of_counts = confusion_matrix(Y_1D, Y_1D_predict)
    # print('matrix_of_counts : ', matrix_of_counts)
    # OR
    matrix_normalized = confusion_matrix(Y_1D, Y_1D_predict, normalize='all')
    # print('matrix_normalized : ', matrix_normalized)

    # Display the confusion matrix
    if plotOUpas == 1!
        if which1 == 'sklearn':
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix_of_counts, display_labels=clf.classes_)
            disp.plot()
            plt.show()
        elif which1 == 'matplotlib':
            import numpy as np
            import matplotlib.pyplot as plt
            %matplotlib inline
            plt.imshow(matrix_of_counts, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar()
            uq_classes = np.unique(Y_1D)
            tick_marks = np.arange(len(uq_classes))
            plt.xticks(tick_marks, uq_classes, rotation=45)
            plt.yticks(tick_marks, uq_classes)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
    
    # ----------------------------
    
    # 3) Classification report : builds a text report showing the main classification metrics
    # Not interested

    # ----------------------------

    # 4) Direct calculation of metrics
    from sklearn import metrics

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    # labels = array-like, default=None

    # pos_label = str or int, default=1 The class to report if average='binary' and the data is binary. 
    # If the data are multiclass or multilabel, this will be ignored

    # average = ['binary', 'micro', 'macro', 'weighted', 'samples', None]
    # This parameter is required for multiclass/multilabel targets. 
    # None : the scores for each class are returned
    # 'binary'  : Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    # 'micro' : Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro' : Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted' : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    # 'samples' : Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    average = 'micro'

    # sample_weight : array-like of shape (n_samples,), default=None
    acc_dircalc = metrics.accuracy_score(Y_1D, Y_1D_predict)
    # print('acc_dircalc : ', acc_dircalc)

    prec_dircalc = metrics.precision_score(Y_1D, Y_1D_predict, average=average)
    # print('prec_dircalc : ', prec_dircalc)

    recall_dircalc = metrics.recall_score(Y_1D, Y_1D_predict, average=average)
    # print('recall_dircalc : ', recall_dircalc)

    f1_dircalc = metrics.f1_score(Y_1D, Y_1D_predict, average=average)
    # print('f1_dircalc : ', f1_dircalc)

    # beta=0.5, 1, 2
    fbeta_dircalc = metrics.fbeta_score(Y_1D, Y_1D_predict, beta=0.5, average=average)
    # print('fbeta_dircalc : ', fbeta_dircalc)
    
    prec_recall_f_dircalc = metrics.precision_recall_fscore_support(Y_1D, Y_1D_predict, beta=0.5, average=average)
    # print('prec_recall_f_dircalc : ', prec_recall_f_dircalc)
    
    # y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
    # y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
    # rocauc_pp_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_pp, average=average) # prediction probability
    # print('rocauc_pp_dircalc : ', rocauc_pp_dircalc)
    # OR
    rocauc_df_dircalc = metrics.roc_auc_score(Y_bin, Y_bin_score, average=average) # decision function 
    # print('rocauc_df_dircalc : ', rocauc_df_dircalc)
    
    rocauc_pp_dircalc = rocauc_df_dircalc

    # ----------------------------
    
    # # 5) Direct calculation of metrics : micro-average ROC curve and ROC area
    # # True Positive Rate (TPR) = TP / (TP + FN) = efficiency (εₛ) to identify the signal (also known as Recall or Sensitivity)
    
    # # False Positive Rate (FPR) = FP / (FP + TN) = inefficiency (ε_B) to reject background
    
    # #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # from sklearn.metrics import roc_curve, auc
    # # Compute ROC curve and ROC area for each class
    # # fpr = dict()
    # # tpr = dict()
    # # roc_auc = dict()
    
    # # # [1] Compute ROC curve and ROC area for each class
    # # for i in range(len(unique_classes)):
        # # fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_score[:, i]) # decision function
        # # # OR
        # # fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], Y_bin_pp[:, i]) # prediction probability (for multi-class)
        
        # # roc_auc[i] = auc(fpr[i], tpr[i])
        
    # # [0] Micro-average ROC curve and ROC area : all the classes together!
    # fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_score.ravel()) # decision function
    # # OR
    # fpr0, tpr0, thresh = roc_curve(Y_bin.ravel(), Y_bin_pp.ravel()) # prediction probability
    
    # roc_auc0 = auc(fpr0, tpr0)

    # # Plot of a ROC curve for a specific class
    # import matplotlib.pyplot as plt
    # plt.figure()
    # lw = 2
    # plt.plot(fpr0, tpr0, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc0)
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # plt.show()
    
    
    value_pack = {}
    
    var_list = ['acc_crossval', 'prec_crossval', 'recall_crossval', 'rocauc_crossval', 'roc_auc_ovo_weighted_crossval', 'acc_dircalc', 'prec_dircalc', 'recall_dircalc', 'f1_dircalc', 'fbeta_dircalc', 'prec_recall_f_dircalc', 'rocauc_pp_dircalc', 'rocauc_df_dircalc']
    var_list_num = [acc_crossval, prec_crossval, recall_crossval, rocauc_crossval, roc_auc_ovo_weighted_crossval, acc_dircalc, prec_dircalc, recall_dircalc, f1_dircalc, fbeta_dircalc, prec_recall_f_dircalc, rocauc_pp_dircalc, rocauc_df_dircalc]
    
    for q in range(len(var_list)):
        value_pack['%s' % (var_list[q])] = var_list_num[q]
    
    return value_pack
    
    
    


# ----------------------------------------------
# Pipelines
# ----------------------------------------------

# Train the model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define preprocessing for numeric columns (normalize them so they're on the same scale)
numeric_features = [0,1,2,3,4,5,6]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (encode the Age column)
categorical_features = [7]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', LogisticRegression(C=1/reg, solver="liblinear"))])


# fit the pipeline to train a logistic regression model on the training set
model = pipeline.fit(X_train, (y_train))
print (model)

# Get predictions from test data
predictions = model.predict(X_test)
y_scores = model.predict_proba(X_test)

# Get evaluation metrics
cm = confusion_matrix(y_test, predictions)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()



# ----------------------------------------------

# LightGBM, CatBoost, scikit-learn and pyspark tree 

# ----------------------------------------------



