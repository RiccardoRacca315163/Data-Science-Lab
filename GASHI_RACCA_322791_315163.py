## IMPORT LIBRARIES ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import random
import time


########################################################################################################################
## WITH THIS ONE WE HAVE BUILT A NEW DF WITH 36 ADDITIONAL COLUMNS ##
## IT TAKES APPROXIMATELY 30 MINUTES TO WRITE THE NEW FILE ##
## INPUT: dataset=dataset to pre-process, feature=the reference feature
## CHECKED ##


def six_pad_df(dataset=None, feature="pmax"):
    particle_df = pd.read_csv(dataset)

    ########################################################################################################################
    ## FILTER OF THE MOST RELEVANT PADS FOR EACH EVENT ##
    df = particle_df.copy()

    df['pmax_1'] = None
    df['pmax_2'] = None
    df['pmax_3'] = None
    df['pmax_4'] = None
    df['pmax_5'] = None
    df['pmax_6'] = None

    df['tmax_1'] = None
    df['tmax_2'] = None
    df['tmax_3'] = None
    df['tmax_4'] = None
    df['tmax_5'] = None
    df['tmax_6'] = None

    df['area_1'] = None
    df['area_2'] = None
    df['area_3'] = None
    df['area_4'] = None
    df['area_5'] = None
    df['area_6'] = None

    df['rms_1'] = None
    df['rms_2'] = None
    df['rms_3'] = None
    df['rms_4'] = None
    df['rms_5'] = None
    df['rms_6'] = None

    df['negpmax_1'] = None
    df['negpmax_2'] = None
    df['negpmax_3'] = None
    df['negpmax_4'] = None
    df['negpmax_5'] = None
    df['negpmax_6'] = None

    df['pad_1'] = None
    df['pad_2'] = None
    df['pad_3'] = None
    df['pad_4'] = None
    df['pad_5'] = None
    df['pad_6'] = None

    for index, row in df.iterrows():
        pmax = []
        tmax = []
        area = []
        rms = []
        negpmax = []

        for col in df.columns:
            if col[:4] == "pmax" and col[-1] == "]":
                pmax.append(row[col])
            elif col[:4] == "tmax" and col[-1] == "]":
                tmax.append(row[col])
            elif col[:4] == "area" and col[-1] == "]":
                area.append(row[col])
            elif col[:3] == "rms" and col[-1] == "]":
                rms.append(row[col])
            elif col[:7] == "negpmax" and col[-1] == "]":
                negpmax.append(row[col])

        name_list = ["pmax", "tmax", "area", "rms", "negpmax"]
        list_lists = [pmax, tmax, area, rms, negpmax]

        for name, lst in zip(name_list, list_lists):
            if name == feature:
                indices = np.argsort(lst)[-6:]  # <- The feature chosen to sort the others.
                break

        largest_pmax = [pmax[i] for i in indices]
        largest_negpmax = [negpmax[i] for i in indices]
        largest_tmax = [tmax[i] for i in indices]
        largest_area = [area[i] for i in indices]
        largest_rms = [rms[i] for i in indices]

        df.loc[index, ["pmax_1", "pmax_2", "pmax_3", "pmax_4", "pmax_5", "pmax_6"]] = largest_pmax
        df.loc[index, ["tmax_1", "tmax_2", "tmax_3", "tmax_4", "tmax_5", "tmax_6"]] = largest_tmax
        df.loc[index, ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]] = largest_area
        df.loc[index, ["rms_1", "rms_2", "rms_3", "rms_4", "rms_5", "rms_6"]] = largest_rms
        df.loc[
            index, ["negpmax_1", "negpmax_2", "negpmax_3", "negpmax_4", "negpmax_5", "negpmax_6"]] = largest_negpmax
        df.loc[index, ["pad_1", "pad_2", "pad_3", "pad_4", "pad_5", "pad_6"]] = indices

        print(f"row {index} finished :)")

    df.to_csv(f"{dataset[0:-4]}_{feature}", index=False, header=True)


# six_pad_df(dataset="development.csv", feature="negpmax")

########################################################################################################################
## THIS GIVES US THE IMPORTANCE OF THE FEATURES AND LEADS US TO CHOOSING THE MOST IMPORTANT IN ORDER TO DETECT THE NOISY PADS ##
## IT RUNS IMMEDIATELY ##
## INPUT: dataset=dataset to pre-process, done in six_pad_df
## OUTPUT: the feature importance of the features chosen in six_pad_df
## CHECKED ##


def feature_importance(dataset=None):
    ####################################################################################################################
    ## EVALUATION OF THE FEATURE ##

    ####################################################################################################################
    ## LOADING THE RANKED DATA ##
    particle_df = pd.read_csv(dataset)  # <- this is only for the development dataset.

    ####################################################################################################################
    ## COMPUTING THE MODE ## <- for each group we have the same mode, consisting of 6 pad add at the end.

    grouped_df = particle_df.groupby(['x', 'y'])
    for ky, group_df in grouped_df:
        for i in range(6):
            moda = group_df[f"pad_{i + 1}"].mode().iloc[0]  # Because mode() gives back a list.
            mask = (particle_df['x'] == group_df['x'].iloc[0]) & (particle_df['y'] == group_df['y'].iloc[0])
            particle_df.loc[mask, f"pad_{i + 1}_mode"] = moda

    ####################################################################################################################
    ## MEASURE OF GOODNESS FOR THE FEATURE ## <- We measure the goodness of a features by computing the number of times the mode is different from the actual data.

    diff_group = []
    for ky, group_df in grouped_df:
        diff_pos = []
        for _, row in group_df.iterrows():
            diff = [0] * 6
            for i in range(6):
                if row[f"pad_{i + 1}"] != row[f"pad_{i + 1}_mode"]:
                    diff[
                        i] += 1  # Maybe different weights can be applied, e.g. the error on the first pad is worse than on the last.
            diff_pos.append(diff)

        print("ky: ", ky, ", diff: ", diff_pos)
        diff_pos = np.array(diff_pos)

        mean_diff_pos = diff_pos.mean(axis=0)
        print(mean_diff_pos)
        diff_group.append(mean_diff_pos)

    print(diff_group)

    diff_group = np.array(diff_group)
    diff_mean = diff_group.mean(axis=0)
    print(diff_mean)
    mean_diff_mean = sum(diff_mean) / len(diff_mean)
    print("Importance of the feature: ", 1 / mean_diff_mean)

    return 1 / mean_diff_mean


# feature_importance(dataset="development_negpmax")

########################################################################################################################
## THE NOISY PADS HAVE BEEN DETECTED ##
## INPUT: feature=the feature to be considered in the distribution analysis, plot=if you want plot select true
## CHECKED ##


def detecting_noisy_sensors(feature="pmax", plot=False):
    particle_df = pd.read_csv("development.csv")

    stat_group = []
    for i in range(18):
        series1 = particle_df[f"{feature}[{i}]"]
        for j in range(18):
            series2 = particle_df[f"{feature}[{j}]"]

            if plot:
                plt.hist(series1, bins=50, density=True, alpha=0.5, color='blue', label=f"{feature}[{i}]")
                plt.hist(series2, bins=50, density=True, alpha=0.5, color='orange', label=f"{feature}[{j}]")

                plt.title('Kernel Density Plot')
                plt.xlabel('Values')
                plt.ylabel('Density')
                plt.legend()

                plt.show()

            from scipy.stats import ks_2samp

            statistic, _ = ks_2samp(series1, series2)

        stat_group.append(statistic)

    stat_group = np.array(stat_group)
    stat_group_mean = stat_group.mean(axis=1)  # <- mean along the rows

    indices_of_highest_values = np.argsort(stat_group_mean)[-6:]

    return indices_of_highest_values  # <- We select the "outliers" indeces w.r.t. the feature.


########################################################################################################################
## THIS FUNCTION BUILDS THE TRAINING SET AND THE TEST SET FOLLOWING THE PRE-PROCESSING SPECIFIED ##
## INPUT: m=number of sample for each "class", features=[list of features to be selected, for example if just pmax is
## selected only the columns containing pmax will be considered].
## CHECKED ##


def bulding_train_and_test(m=1, features=None):
    ########################################################################################################################
    ## LOAD OF THE FILE ##

    particle_train_df = pd.read_csv('development.csv')
    particle_test_df = pd.read_csv('evaluation.csv')

    ########################################################################################################################
    ## DISCARDING OF SENSORS ##

    to_drop_indices = [0, 7, 12, 15, 16, 17]

    columns = particle_train_df.columns.tolist()
    to_drop = [col for col in columns if any(str(num) in col and "10" not in col for num in to_drop_indices)]

    particle_train_df = particle_train_df.drop(columns=to_drop, axis=1)
    particle_test_df = particle_test_df.drop(columns=to_drop, axis=1)

    ########################################################################################################################
    ## SELECTING A NUMBER m OF ROWS FOR EACH GROUP, OBTAINING A REDUCED DATAFRAME ##

    particle_train_df = particle_train_df.groupby(['x', 'y'], group_keys=False).apply(lambda x: x.sample(m))

    ########################################################################################################################
    ## SHUFFLING THE TRAIN DF, THEY ARE STORED IN A SORTED WAY ##

    particle_train_df = particle_train_df.sample(frac=1, random_state=312123)
    particle_train_df.reset_index(drop=True, inplace=True)

    ########################################################################################################################
    ## SAVING THE TARGET VARIABLES APART AND REMOVING ID FROM EVALUATION ##

    columns = ['x', 'y']
    target_train = pd.DataFrame(columns=columns)

    target_train['x'] = particle_train_df['x']
    target_train['y'] = particle_train_df['y']

    target_train = np.array(target_train)

    particle_train_df = particle_train_df.drop(columns=['x', 'y'])
    particle_test_df = particle_test_df.drop(columns=["Id"])

    ########################################################################################################################
    ## FEATURES SELECTION: ##

    if features is not None:
        for col in particle_train_df.columns:
            if col[:-3] not in features and col[:-4] not in features:
                particle_train_df = particle_train_df.drop(col, axis=1)
                particle_test_df = particle_test_df.drop(col, axis=1)

    ########################################################################################################################
    ## TO NUMPY ARRAY ##

    print(particle_train_df.columns)
    X_train = np.array(particle_train_df)
    X_test = np.array(particle_test_df)

    return X_train, X_test, target_train


########################################################################################################################
## WE MISS UP TO THIS POINT JUST TO JUSTIFY THE FEATURES SELECTED, IT IS REASONABLE TO DO SO GOING THROUGH A REGRESSION MODEL ##

########################################################################################################################
## THIS FUNCTION EMULATES THE METRIC USED IN THE CHALLENGE ##
## CHECKED ##


def compute_distance(target_pred=None, target_true=None, return_back=False):
    distance = 0
    n = len(target_pred)
    for index, _ in enumerate(target_pred):
        distance += np.sqrt((float(target_true[index, 0]) - float(target_pred[index, 0])) ** 2 + (
                float(target_true[index, 1]) - float(target_pred[index, 1])) ** 2)

    distance = distance / n

    print(f"Mean of distances: {distance}")

    if return_back:
        return distance

########################################################################################################################
## THIS FUNCTION JUST WRITES THE FILE AS REQUESTED BY THE TASK ##
# INPUT: file_path= the file uploaded by the teacher from where I check the structure, file_path2=the file I write, target_test_pred=The prediction
# In my code the file uploaded bby teacher was called: "sample_submission_true.csv", while the one I wrote was "sample_submission.csv".
# OUTPUT: NO OUTPUT, IT WRITES DIRECTLY ONTO THE FILE.
## CHECKED ##


def writing_file(file_path=None, file_path2=None, target_test_pred=None):
    df = pd.read_csv(file_path)

    if len(target_test_pred) == len(df):
        df['Predicted'] = ['|'.join(map(str, row)) for row in target_test_pred]

        df.to_csv(file_path2, index=False, header=True)
        print("Column 'Predicted' updated in the CSV file successfully!")
    else:
        print("The length of target_test_pred does not match the length of the 'Predicted' column in the DataFrame.")

########################################################################################################################
## IT IS NOT AN ACTUAL FUNCTION, JUST KEEPS TIGHT THE PIPELINES TRIED ##


def functions_list():
    ########################################################################################################################
    ## MODELS LIST ##

    ########################################################################################################################
    ## RANDOM FOREST REGRESSOR ##
    """
    from sklearn.ensemble import RandomForestRegressor

    # n_estimators : The number of trees in the forest.
    # criterion : “squared_error”, “absolute_error”, “friedman_mse”, “poisson”, default=”squared_error”
    # max_features : “sqrt”, “log2”, None, {int or float}, default=1.0
    # oob_score : {bool or callable}, default=False
    # n_jobsint : default=None

    pipeline = make_pipeline(
        MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=random_seed))
    )
    """
    ########################################################################################################################
    ## SVR ##
    """
    from sklearn.svm import SVR

    # kernel : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # C : Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    # epsilon : Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Must be non-negative.

    kernel = "sigmoid"
    pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(SVR(kernel=kernel, epsilon=1, C=20, gamma="scale"))
    )
    """
    ########################################################################################################################
    ## MULTI-LAYER NEURAL NETWORK ##
    """
    from sklearn.neural_network import MLPRegressor

    # hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    # activation : ‘identity’, ‘logistic’, ‘tanh’, ‘relu’, default=’relu’
    # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
    # max_iter : Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.

    pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(40,), activation="logistic", learning_rate="constant", max_iter=5000, random_state=random_seed))
    )
    """
    ########################################################################################################################
    ## REGRESSIONE LINEARE ##
    """
    from sklearn.linear_model import LinearRegression

    # fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

    pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(LinearRegression(fit_intercept=True))
    )
    """
    ########################################################################################################################
    ## REGRESSIONE LASSO ##
    """
    from sklearn.linear_model import Lasso

    # alpha: Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float
    # fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

    pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(Lasso(alpha=200, fit_intercept=True))
    )
    """
    ########################################################################################################################
    ## REGRESSIONE RIDGE ##
    """
    from sklearn.linear_model import Ridge

    # alpha: Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative float
    # fit_intercept: Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations

    pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(Ridge(alpha=2000, fit_intercept=True))
    )
    """

########################################################################################################################
## THIS IS THE FUNCTION USED TO PERFORM THE MODEL EVALUATION ##
# INPUT: m=1, features=None, pipeline=None #
# OUTPUT: target_pred, distance #
## CHECKED ##


def model_evaluation(attributes_double=None):
    m = attributes_double[0]
    features = attributes_double[1]

    X, _, target = bulding_train_and_test(m=m, features=features)

    ####################################################################################################################
    ## TRAIN AND TEST SPLIT ##

    frac = 0.2

    X, X_t, target, target_t = train_test_split(X, target, test_size=frac)  # SHUFFLE HAS ALREADY BEEN DONE

    """all_indices = list(range(len(X)))
    set_all = set(all_indices)

    test_indices = random.sample(range(len(X)), round(len(X) * frac))
    set_test = set(test_indices)

    set_train = set_all.difference(set_test)
    train_indices = list(set_train)

    X_t = X[test_indices]
    X = X[train_indices]

    target_t = target[test_indices]
    target = target[train_indices]"""

    ####################################################################################################################
    ## REGRESSOR MODEL ##

    pipeline = attributes_double[2]

    pipeline.fit(X, target)

    target_pred = pipeline.predict(X_t)

    print("This is the target:\n", target_pred)

    ################################################################################################################
    ## COMPUTE THE DISTANCE ##

    distance = compute_distance(target_pred=target_pred, target_true=target_t, return_back=True)

    return target_pred, distance

########################################################################################################################
## THIS FUNCTION WRITES THE FILE AFTER HAVING APPLIED THE MODEL SELECTED ONTO THE EVALUATION DATASET ##
# INPUT: m=1, features=None, pipeline=None #
# DOUBLE: m=1, features=None, pipeline=None #
## CHECKED ##


def code_to_file_model(attributes_double=None):
    ####################################################################################################################
    # PRE-PROCESSING #

    m = attributes_double[0]
    features = attributes_double[1]
    X_train, X_test, target_train = bulding_train_and_test(m=m, features=features)

    ####################################################################################################################
    ## REGRESSOR MODEL ##

    pipeline = attributes_double[2]

    # Fit the pipeline
    pipeline.fit(X_train, target_train)

    # Evaluate the performance of the model on the test set
    target_test_pred = pipeline.predict(X_test)

    print("This is the target:\n", target_test_pred)

    ####################################################################################################################
    ## WRITING THE FILE ##

    file_path = "sample_submission_true.csv"
    file_path_2 = "sample_submission.csv"

    writing_file(file_path=file_path, file_path2=file_path_2, target_test_pred=target_test_pred)

########################################################################################################################
## THIS FUNCTION RETURNS THE EVALUATION OF THE ENSAMBLE PIPELINES SELECTED ##
# INPUT:
# t: size of the sampling for the test set.
# LIST OF: [DOUBLE: m=1, features=None, pipeline=None]
# k: coefficient acting on weights.
## CHECKED ##


def ensemble_evaluation(t=10, list_attributes_double=None, k=1):
    ########################################################################################################################
    ## LOAD OF THE FILE ##

    particle_df = pd.read_csv('development.csv')

    ################################################################################################################
    ## DISCARDING THE NOISY-SENSORS ##

    to_drop_indices = [0, 7, 12, 15, 16, 17]

    columns = particle_df.columns.tolist()
    to_drop = [col for col in columns if any(str(num) in col and "10" not in col for num in to_drop_indices)]

    particle_df = particle_df.drop(columns=to_drop, axis=1)

    ####################################################################################################################
    ## TRAIN+VALIDATION AND TEST SPLIT ##

    print("Length particle_df before test_df:", len(particle_df))
    test_df = particle_df.groupby(['x', 'y'], group_keys=False).apply(lambda x: x.sample(t))

    test_df_index = test_df.index
    particle_df = particle_df.drop(test_df_index)
    print("Length particle_df after test_df:", len(particle_df))

    ####################################################################################################################
    ## STORING THE TARGET_TEST DF FOR THE TEST ##

    columns = ['x', 'y']
    target_test = pd.DataFrame(columns=columns)

    target_test['x'] = test_df['x']
    target_test['y'] = test_df['y']

    target_test = np.array(target_test)
    test_df = test_df.drop(columns=['x', 'y'])
    test_df_2 = test_df.copy()  # AUXILIAR VARIABLE TO KEEP TEST_DF INDEPENDENT FROM TRAIN_DF

    ####################################################################################################################
    ## JUST USING TRAIN-VALIDATION ##

    list_inv_distances = []
    list_pipelines_fitted = []

    for att_double in list_attributes_double:
        test_df = test_df_2.copy()

        m = att_double[0]
        train_valid_df = particle_df.groupby(['x', 'y'], group_keys=False).apply(lambda x: x.sample(m))

        ################################################################################################################
        ## SAVING THE TARGET DF FOR THE TRAIN-VALIDATION ##

        target_train_valid = pd.DataFrame(columns=columns)

        target_train_valid['x'] = train_valid_df['x']
        target_train_valid['y'] = train_valid_df['y']

        target_train_valid = np.array(target_train_valid)

        train_valid_df = train_valid_df.drop(columns=['x', 'y'])

        ################################################################################################################
        ## SELECTING THE FEATURES ##

        features = att_double[1]

        if features is not None:
            for col in train_valid_df.columns:
                if col[:-3] not in features and col[:-4] not in features:
                    train_valid_df = train_valid_df.drop(col, axis=1)
                    test_df = test_df.drop(col, axis=1)

        ################################################################################################################
        ## TO NUMPY ARRAY ##

        print(particle_df.columns)
        X_train_valid = np.array(train_valid_df)
        X_test = np.array(test_df)

        ################################################################################################################
        ## TRAIN AND VALIDATION SPLIT ##

        frac = 0.2

        all_indices = list(range(len(X_train_valid)))
        set_all = set(all_indices)

        valid_indices = random.sample(range(len(X_train_valid)), round(len(X_train_valid) * frac))
        set_valid = set(valid_indices)

        set_train = set_all.difference(set_valid)
        train_indices = list(set_train)

        X_valid = X_train_valid[valid_indices]
        X_train = X_train_valid[train_indices]

        target_valid = target_train_valid[valid_indices]
        target_train = target_train_valid[train_indices]

        ################################################################################################################
        ## REGRESSOR MODEL ##

        pipeline = att_double[2]

        # Fit the pipeline
        pipeline.fit(X_train, target_train)

        # Evaluate the performance of the model on the test set
        target_valid_pred = pipeline.predict(X_valid)

        ################################################################################################################
        ## COMPUTE THE DISTANCE ##

        distance = compute_distance(target_pred=target_valid_pred, target_true=target_valid, return_back=True)

        ################################################################################################################
        ## SAVING THE ENTITIES FOUND ##

        list_inv_distances.append(1 / (distance**k))
        list_pipelines_fitted.append(pipeline)

    ####################################################################################################################
    ## COMPUTING THE ENSEMBLE PREDICTION ##

    target_test_pred = list_pipelines_fitted[0].predict(X_test) * 0
    print(len(list_pipelines_fitted))
    for index, pipe in enumerate(list_pipelines_fitted):
        predicted = pipe.predict(X_test)
        print("This is the target:\n", predicted)
        print(f"Prediction with double pipeline {index} finished")
        target_test_pred = target_test_pred + predicted * list_inv_distances[index]

    target_test_pred = target_test_pred / sum(list_inv_distances)

    compute_distance(target_pred=target_test_pred, target_true=target_test)

    return list_pipelines_fitted, list_inv_distances

########################################################################################################################
## THIS FUNCTION WRITES THE FILE AFTER HAVING APPLIED THE ENSEMBLE MODEL SELECTED ONTO THE EVALUATION DATASET ##
# INPUT: -> LIST OF: [DOUBLE: m=1, features=None, pipeline=None]
# OUTPUT: NO OUTPUT, IT WRITES DIRECTLY ON THE FILE.
## CHECKED ##


def code_to_file_ensemble(list_attributes_double=None):
    ####################################################################################################################
    ## ENSAMBLE EVALUATION TO OBTAIN THE FITTED PIPELINES AND THE WEIGHTS ##

    list_pipelines_fitted, list_inv_distances = ensemble_evaluation(t=20, list_attributes_double=list_attributes_double)

    ####################################################################################################################
    ## APPLYING ITERATEVELY THE PIPELINES AND CONVEXLY COMBINE THEM TO OBTAIN THE PREDICTION ##

    for index in range(len(list_pipelines_fitted)):

        m = list_attributes_double[index][0]
        features = list_attributes_double[index][1]
        _, X_test, _ = bulding_train_and_test(m=m, features=features)

        if index == 0:
            target_test_pred = list_pipelines_fitted[0].predict(X_test) * 0  # TO MATCH ITS SIZE

        target_test_pred_pipe = list_pipelines_fitted[index].predict(X_test)
        target_test_pred = target_test_pred + target_test_pred_pipe * list_inv_distances[index]

    target_test_pred = target_test_pred / sum(list_inv_distances)

    #########################################################################################################################
    ## WRITING THE FILE ##

    file_path = "sample_submission_true.csv"
    file_path_2 = "sample_submission.csv"

    writing_file(file_path=file_path, file_path2=file_path_2, target_test_pred=target_test_pred)


########################################################################################################################
# THIS FUNCTION RETURNS THE OPTIMAL HYPER-PARAMETERS OF THE SELECTED PIPELINE #
# INPUT: m=1, features=None
# OUTPUT: NO OUTPUT, IT PRINTS THE BEST RESULTS OBTAINED. #
## CHECKED ##


def grid_search_made(m=1, features=None):
    # DEFINITION OF THE CUSTOM SCORE #

    def compute_distance_no_return(target_true, target_pred):
        distance = 0
        n = len(target_pred)
        for index, row in enumerate(target_pred):
            distance += np.sqrt((float(target_true[index, 0]) - float(target_pred[index, 0])) ** 2 + (
                    float(target_true[index, 1]) - float(target_pred[index, 1])) ** 2)

        distance = distance / n

        print(f"Mean of distances: {distance}")

        return distance

    # IN THIS SECTION IT MUST BE IMPORTED ALL IT IS NEEDED #

    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer

    # PRE-PROCESSING TO OBTAIN THE TRAINING SET #

    X_train, _, target_train = bulding_train_and_test(m=m, features=features)

    # PIPELINE TO BUILD #

    pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(SVR())
    )

    # PARAMETERS' SPACE #

    param_grid = {
        'multioutputregressor__estimator__C': [200],
        'multioutputregressor__estimator__kernel': ['rbf'],
        'multioutputregressor__estimator__gamma': ['auto'],
        'multioutputregressor__estimator__epsilon': [0.5]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_scorer(compute_distance_no_return, greater_is_better=False),
        cv=3,
        verbose=2
    )

    grid_search.fit(X_train, target_train)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    print(best_params, best_estimator)


########################################################################################################################
# THIS FUNCTION RETURNS THE OPTIMAL HYPER-PARAMETERS OF THE SELECTED PIPELINE #
# INPUT: m=1, features=None
# OUTPUT: NO OUTPUT, IT PRINTS THE BEST RESULTS OBTAINED. ##
## CHECKED ##


def halving_grid_search_made(m=1, features=None):
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.metrics import make_scorer

    def compute_distance_no_return(target_true, target_pred):
        distance = 0
        n = len(target_pred)
        for index, row in enumerate(target_pred):
            distance += np.sqrt((float(target_true[index, 0]) - float(target_pred[index, 0])) ** 2 + (
                    float(target_true[index, 1]) - float(target_pred[index, 1])) ** 2)

        distance = distance / n

        print(f"Mean of distances: {distance}")

        return distance

    X_train, _, target_train = bulding_train_and_test(m=m, features=features)

    from sklearn.neural_network import MLPRegressor
    pipeline = make_pipeline(
        MultiOutputRegressor(MLPRegressor(max_iter=10000))
    )

    param_grid = {
        'multioutputregressor__estimator__hidden_layer_sizes': [(150,), (50, 50), (50, 100), (100, 50), (100, 100), (50, 50, 50), (50, 100, 50), (100, 50, 50), (50, 50, 100)],
        'multioutputregressor__estimator__activation': ['logistic'],
        'multioutputregressor__estimator__alpha': [0.00001, 0.0001, 0.001],
        'multioutputregressor__estimator__learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    scorer = make_scorer(compute_distance_no_return, greater_is_better=False)

    halving_grid_search = HalvingGridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        factor=3,
        verbose=3,
        random_state=21235080
    )

    halving_grid_search.fit(X_train, target_train)

    best_params = halving_grid_search.best_params_
    best_estimator = halving_grid_search.best_estimator_

    print(best_params, best_estimator)


########################################################################################################################
## NOW IT WILL BE LISTED A SERIES OF "TRIES" FUNCTIONS, JUST TO CALL THE ABOVE-DEFINED ONES. ##

########################################################################################################################
## FUNCTION TO EVALUATE A MODEL ##
## CHECKED ##


def tries_model_evaluation():
    ####################################################################################################################
    ## PIPELINE ##
    from sklearn.neural_network import MLPRegressor
    pipeline = make_pipeline(
        MultiOutputRegressor(MLPRegressor(activation='logistic',
                                          alpha=1e-05,
                                          hidden_layer_sizes=(250,),
                                          learning_rate='constant',
                                          max_iter=5000)))

    ####################################################################################################################
    ## DOUBLE TARGET EVALUATION ##

    start_time = time.time()

    m = 80
    attributes_double = [m, ["pmax", "negpmax"], pipeline]
    _, _ = model_evaluation(attributes_double=attributes_double)

    print("--- %s seconds ---" % (time.time() - start_time))


########################################################################################################################
## FUNCTION TO EVALUATE AN ENSEMBLE MODEL ##
## CHECKED ##

def tries_with_ensemble_evaluation():
    ####################################################################################################################
    # DOUBLE PIPELINES #

    # 0 - RANDOM FOREST #
    from sklearn.ensemble import RandomForestRegressor

    # n_estimators : The number of trees in the forest.
    # criterion : “squared_error”, “absolute_error”, “friedman_mse”, “poisson”, default="squared_error"
    # max_features : “sqrt”, “log2”, None, {int or float}, default=1.0
    # oob_score : {bool or callable}, default=False
    # n_jobsint : default=None

    pipeline_0 = make_pipeline(
        MultiOutputRegressor(
            RandomForestRegressor(n_estimators=30, max_features="sqrt"))
    )

    ########################################################################################################################
    # 1 - SVR #

    from sklearn.svm import SVR

    # kernel : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # C : Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    # epsilon : Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Must be non-negative.

    kernel = "rbf"
    pipeline_1 = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(SVR(kernel=kernel, epsilon=0.5, C=100))
    )

    ########################################################################################################################
    # 2 - ML - NN #

    from sklearn.neural_network import MLPRegressor

    # hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    # activation : ‘identity’, ‘logistic’, ‘tanh’, ‘relu’, default="relu"
    # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default="constant"
    # max_iter : Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.

    pipeline_2 = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(
            MLPRegressor(hidden_layer_sizes=(100,), activation="logistic", learning_rate="constant", max_iter=5000))
    )

    ########################################################################################################################
    # PIPELINES #
    #
    # INPUT:
    # t: size of the sampling for the test set.
    # DOUBLE: m=1, features=None, pipeline=None

    list_attributes_double = []

    m_rf = 30
    list_attributes_double.append([m_rf, ["pmax", "negpmax"], pipeline_0])
    list_attributes_double.append([m_rf, ["pmax", "negpmax"], pipeline_0])
    list_attributes_double.append([m_rf, ["pmax", "negpmax"], pipeline_0])

    m_svr = 6
    list_attributes_double.append([m_svr, ["pmax", "negpmax"], pipeline_1])
    list_attributes_double.append([m_svr, ["pmax", "negpmax"], pipeline_1])
    list_attributes_double.append([m_svr, ["pmax", "negpmax"], pipeline_1])

    m_ml_nn = 8
    list_attributes_double.append([m_ml_nn, ["pmax", "negpmax"], pipeline_2])
    list_attributes_double.append([m_ml_nn, ["pmax", "negpmax"], pipeline_2])
    list_attributes_double.append([m_ml_nn, ["pmax", "negpmax"], pipeline_2])

    #########################################################################################################################
    ## ENSEMBLE EVALUATION ##

    start_time = time.time()

    _, list_inv_distances = ensemble_evaluation(t=20, list_attributes_double=list_attributes_double)
    print("list_inv_distances:", list_inv_distances)

    print("--- %s seconds ---" % (time.time() - start_time))


########################################################################################################################
## FUNCTION TO WRITE THE FILE AFTER HAVING APPLIED THE MODEL ##
## CHECKED ##


def tries_code_to_file_model():
    ####################################################################################################################
    ## PIPELINE ##
    ## PIPELINE ##
    from sklearn.neural_network import MLPRegressor
    pipeline = make_pipeline(
        MultiOutputRegressor(MLPRegressor(activation='logistic',
                                          alpha=1e-05,
                                          hidden_layer_sizes=(200, 200),
                                          learning_rate='constant',
                                          max_iter=5000)))

    ####################################################################################################################
    ## DOUBLE - TO - CODE  ##

    start_time = time.time()

    m = 100
    attributes_double = [m, ["pmax", "negpmax"], pipeline]
    code_to_file_model(attributes_double=attributes_double)

    print("--- %s seconds ---" % (time.time() - start_time))


########################################################################################################################
## FUNCTION TO WRITE THE FILE AFTER HAVING APPLIED THE ENSEMBLE MODEL ##
## CHECKED ##

def tries_code_to_file_ensemble():
    ####################################################################################################################
    # DOUBLE PIPELINES #

    # 0 - RANDOM FOREST #
    from sklearn.ensemble import RandomForestRegressor

    # n_estimators : The number of trees in the forest.
    # criterion : “squared_error”, “absolute_error”, “friedman_mse”, “poisson”, default="squared_error"
    # max_features : “sqrt”, “log2”, None, {int or float}, default=1.0
    # oob_score : {bool or callable}, default=False
    # n_jobsint : default=None

    pipeline_0 = make_pipeline(
        MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_features="sqrt"))
    )

    ###################################################################################################################
    # 1 - SVR #

    from sklearn.svm import SVR

    # kernel : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # C : Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    # epsilon : Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Must be non-negative.

    kernel = "rbf"
    pipeline_1 = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(SVR(kernel=kernel, epsilon=0.5, C=200))
    )

    ####################################################################################################################
    # 2 - ML - NN #

    from sklearn.neural_network import MLPRegressor

    # hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    # activation : ‘identity’, ‘logistic’, ‘tanh’, ‘relu’, default="relu"
    # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default="constant"
    # max_iter : Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.

    pipeline_2 = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(
            MLPRegressor(hidden_layer_sizes=(200,), activation="logistic", learning_rate="constant", max_iter=5000))
    )

    ####################################################################################################################
    # PIPELINES #
    #
    # INPUT:
    # t: size of the sampling for the test set.
    # DOUBLE: m=1, features=None, pipeline=None

    list_attributes_double = []

    m_rf = 100
    list_attributes_double.append([m_rf, ["pmax", "negpmax"], pipeline_0])

    m_svr = 30
    list_attributes_double.append([m_svr, ["pmax", "negpmax"], pipeline_1])

    m_ml_nn = 100
    list_attributes_double.append([m_ml_nn, ["pmax", "negpmax"], pipeline_2])

    ####################################################################################################################

    start_time = time.time()

    code_to_file_ensemble(list_attributes_double=list_attributes_double)

    print("--- %s seconds ---" % (time.time() - start_time))


########################################################################################################################
## THIS FUNCTION APPROXIMATE THE RESULTS IN ORDER TO FOLLOW THE SCHEME PROVIDED BY THE DEVELOPMENT TARGET VARIABLES.
## IT MUST BE USED CAREFULLY, IN THE TASK PRESENTED IT IMPROVES SIGNIFICALLY THE RESULTS, SINCE PROBABLY THE
## DATA ARE TAKEN UNDER THE SAME CONDITIONS. BUT THIS PRACTICALLY REDUCE THE REGRESSION PROBLEM INTO A CLASSIFICATION ONE.
## THEREFORE IT APPLICABILITY IN FUTURE OCCASIONS CAN LEAD TO WORSE RESULTS.


def approximation():
    df = pd.read_csv('development.csv')

    unique_x = np.sort(df['x'].unique())
    unique_y = np.sort(df['y'].unique())

    df2 = pd.read_csv("sample_submission.csv")

    df2[['coord_x', 'coord_y']] = df2['Predicted'].str.split('|', expand=True)

    df2 = df2.drop(columns=['Id', 'Predicted'])

    df2['coord_x'] = pd.to_numeric(df2['coord_x'])
    df2['coord_y'] = pd.to_numeric(df2['coord_y'])

    print(df2)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    df2['coord_x'] = df2['coord_x'].apply(lambda x: find_nearest(unique_x, x))
    df2['coord_y'] = df2['coord_y'].apply(lambda y: find_nearest(unique_y, y))

    target_test_pred = np.array(df2)
    print(target_test_pred)

    file_path = "sample_submission_true.csv"
    file_path2 = "sample_submission.csv"
    writing_file(file_path=file_path, file_path2=file_path2, target_test_pred=target_test_pred)


########################################################################################################################

# tries_code_to_file_model()
# tries_model_evaluation()
# halving_grid_search_made(m=6, features=["pmax", "negpmax"])
