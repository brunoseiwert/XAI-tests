import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency
from typing import Union
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, fbeta_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
import warnings

#set_config(transform_output="pandas")
seed = 42


def load_dataset(path: str, columns: list = None) -> pd.DataFrame:
    """ load the dataset
    Parameters:
    ------------------
    path: str
        full path of the dataset
    column:list
        List of the header of the dataset, default is None
        
    Returns:
    ------------------
    data: pd.DataFrame
        the dataset
    """

    data = None
    if columns is None:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, sep=" ", header=None)
        # check if the length of the column is equal to the number of the columns
        assert len(columns) == data.shape[1]
        data.columns = columns
    return data


def check_duplicates(data: pd.DataFrame):
    """ drops the duplicates in dataset
    Parameters:
    ------------------
    data: pd.DataFrame
        dataset to drop duplicates if it has
    """

    n_duplicates = np.sum(data.duplicated())
    if n_duplicates == 0:
        statement = "no duplicates"
    else:
        statement = str(n_duplicates) + " duplicates."
    data.drop_duplicates()
    print(f'There exists : {statement}')


def output_distribution(dataframe: pd.DataFrame, output: str = None):
    """ calculates the distribution of the output class
    Parameters:
    ------------------
    dataframe: pd.DataFrame
        dataset to calculate the its distribution
    Returns:
    ------------------
    data: pd.Series
        the distribution of the output class
    """
    if output is None:
        output = dataframe.columns.tolist()[-1]
    counter = dataframe[output].value_counts(normalize=True) * 100
    return counter


def get_numerical_columns(dataframe: pd.DataFrame, output: str = None) -> list:
    """ returns the numerical columns of the dataset
    Parameters:
    ------------------
    dataframe: pd.DataFrame
        dataset
    output:str
        output class
        
    Returns:
    ------------------
    list
        list of the columns whose type is either int64 or float64
    """
    columns = dataframe.select_dtypes(['int64', 'float64', 'float32', 'int32']).columns.tolist()
    if output is not None:
        columns.remove(output)
    return columns


def get_categorical_columns(dataframe: pd.DataFrame, output: str = None) -> list:
    """ returns the categorical columns of the dataset
    Parameters:
    ------------------
    dataframe: pd.DataFrame
        dataset 
    output:str
        output class
        
    Returns:
    ------------------
    columns : list
        list of the columns whose type is either object or bool
    """
    columns = dataframe.select_dtypes(['object', 'bool', 'category']).columns.tolist()
    if output is not None:
        columns.remove(output)
    return columns


def print_list(title: str, listToPrint: list):
    """ print each element of the list
    Parameters:
    ------------------
    title: str
        title of the list
    listToPrint: list
        list to print its each element
    """
    print(f'{title} \n')
    for i, feature in enumerate(listToPrint):
        print(f'\t {i + 1}. {feature} \n')


def plot_hist(dataframe: pd.DataFrame):
    """ creates histogram for numerical variables
    Parameters:
    ------------------
    dataframe: pd.DataFrame
        dataset
    """
    num_columns = get_numerical_columns(dataframe)
    num_data = dataframe[num_columns]
    ax = num_data.hist(figsize=(18, 10))
    plt.show()


def check_missings(dataframe: pd.DataFrame) -> list:
    """ Checks missing values, prints missing state
    of variables
    Parameters:
    ------------------
    dataframe: pd.DataFrame
        dataset to split into features and output
        
    Returns:
    ------------------
    null_columns: list
        list of attributes which has missing value
    """
    null_columns = list()
    null_items = np.sum(dataframe.isnull())
    for k, v in null_items.items():
        if v != 0:
            null_columns.append(k)
    if len(null_columns) == 0:
        print(f'There are no missing attributes')
    else:
        title = "Attributes that have missing"
        print_list(title, null_columns)
    return null_columns


def test_anova(df_X: pd.DataFrame, df_Y: pd.DataFrame, num_cols: list) -> list:
    """ Tests numerical columns with categorical target data by anova test to see
    importance of features

    Parameters:
    ------------------
    df_X: pd.DataFrame
        dataset to test columns 
    df_Y: pd.DataFrame
        output class
    num_cols:list
        numerical columns
    Returns:
    ------------------
    selectedFeatures: list
        list of features which shows the significant importance for output class
    """
    dataframe = pd.concat([df_X, df_Y], axis=1)
    target = df_Y.name
    selected_features = list()
    for col in num_cols:
        group_list = dataframe.groupby(target)[col].apply(list)
        result = f_oneway(*group_list)
        if result[1] < 0.05:
            selected_features.append(col)
    #         print(f'{col} p-value: {result[1]}')
    return selected_features


def test_chiSq(df_X: pd.DataFrame, df_Y: pd.DataFrame, cat_cols: list) -> list:
    """ Tests categorical columns with categorical target data by Chi-Square test to see
    importance of features

    Parameters:
    ------------------
    df_X: pd.DataFrame
        dataset to test columns 
    df_Y: pd.DataFrame
        output class
    cat_cols:list
        categorical columns
    Returns:
    ------------------
    selected_features: list
        a list of features which shows the significant importance for output class
    """
    dataframe = pd.concat([df_X, df_Y], axis=1)
    target = df_Y.name
    selected_features = list()
    for col in cat_cols:
        cross_tab = pd.crosstab(dataframe[target], dataframe[col])
        result = chi2_contingency(cross_tab)
        if result[1] < 0.05:
            selected_features.append(col)
    return selected_features


def eliminate_feats(df_x: pd.DataFrame, df_y: pd.DataFrame, num_cols: list = None, cat_cols: list = None):
    """ Finds the relevant features for model's output via statistical tests. One way anova test is used to 
    extract relation between continuous features and target class, wheras Chi-square test is applied to find 
    relation between categorical features and target class. 

    Parameters:
    ------------------
    df_x: pd.DataFrame
        dataset
    df_y:pd.DataFrame
        target class
    num_cols:list
        continuous attributes
    cat_cols:list
        categorical attributes

    Returns:
    ------------------
    pd.Dataframe, list, list
    df_x[selected]: pd.DataFrame
        dataset with selected attributes
    sel_num_cols: list



    """
    sel_num_cols = None
    sel_cat_cols = None
    selected = list()
    if num_cols is not None:
        sel_num_cols = test_anova(df_x, df_y, num_cols)
        selected += sel_num_cols
    if cat_cols is not None:
        sel_cat_cols = test_chiSq(df_x, df_y, cat_cols)
        selected += sel_cat_cols
    return df_x[selected], sel_num_cols, sel_cat_cols


def change_features(df: pd.DataFrame, dict_set: dict):
    """ Changes the features to make more readable categorical attributes
    Parameters:
    ------------------
    df: pd.DataFrame
        dataset to change features
    dict_set = dict
        dictionaries that includes attributes and new features that should be changed.
        
    Returns:
    ------------------
    df_cpy : pd.DataFrame
        new dataframe whose features are changed according to dictionary
    """
    df_cpy = df.copy()
    cat_cols = get_categorical_columns(df_cpy)
    for col in cat_cols:
        df_cpy[col] = df_cpy[col].map(dict_set[col])
    return df_cpy


def split_IO(dataframe: pd.DataFrame, output: str) -> Union[pd.DataFrame, pd.DataFrame]:
    """ splits the dataset into features and output class
    Parameters:
    ------------------
    dataframe: pd.DataFrame
        dataset to split into features and output
        
    Returns:
    ------------------
    X: pd.DataFrame
        features
    Y: pd.DataFrame
        output class
    """

    X, Y = dataframe.drop(output, axis=1), dataframe[output]
    return X, Y


def fbetascore(y_true: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray]):
    """computes F Beta score, that is harmonic mean of precision 
    and recall and gives more weights to the recall
    Parameters:
    ------------------
    y_true: Union[pd.DataFrame, np.ndarray]
        real class labels
    y_pred: Union[pd.DataFrame, np.ndarray]
        predicted class labels
        
    Returns:
    ------------------
    scores:callable
         Callable object that returns a scalar score

    """
    return fbeta_score(y_true, y_pred, beta=2)


def evaluate_model(X_data: pd.DataFrame, y_data: pd.DataFrame, model, scoring):
    """ evaluates the model according to the given evaluation strategy
    Parameters:
    ------------------
    X_data : pd.DataFrame
        dataset to evaluate model
    y_data: pd.DataFrame
        output class 
    model: model object implementing fit
        object to use to fit the data
    scoring : str
        metrics to evaluate
        
    Returns:
    ------------------
    scores:callable
         Callable object that returns a scalar score
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
    scores = cross_val_score(model, X_data.values, y_data.values.ravel(), scoring=scoring, cv=cv, n_jobs=-1)
    return scores


def evaluate_all_models(X_data: pd.DataFrame, Y_data: pd.DataFrame, model_dict: dict,
                        scoring=make_scorer(fbetascore), oversampling: bool = False):
    """ evaluates all models according to the given evaluation strategy
    Parameters:
    ------------------
    X_data : pd.DataFrame
        dataset to evaluate model
    y_data: pd.DataFrame
        output class 
    model_dict: dict
        objects and their names to use to fit the data 
        
    Returns:
    ------------------
    scores:pd.DataFrame
        a scalar score for all models
    """
    X_data_n = X_data.copy()
    Y_data_n = Y_data.copy()
    scores = pd.DataFrame()

    if oversampling:
        sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
        X_data_n, Y_data_n = sampling.fit_resample(X_data, Y_data)
    for name, model in model_dict.items():
        print(f'{name} is working...')
        score = evaluate_model(X_data_n, Y_data_n, model, scoring)
        scores[name] = score
    return scores


def create_pipeline(ohe_cols: list = None, ord_cols: list = None, num_cols: list = None, is_std_scaler: bool = False):
    """ creates pipelines
        Parameters:
        ------------------
        ohe_cols : list
            feature list for encoding by one hot encoder
        ord_cols : list
            feature list for encoding by ordinal encoder
        num_cols : list
            feature list for scaling
        ist_std_scaler : bool
            if yes, standard scaler will be used, not Min Max Scaler

        Returns:
        ------------------
        pipeline : callable
            callable object with type of Pipeline
        """
    cols = ['one', 'ord', 'num']
    transformers = dict(zip(cols, [None] * len(cols)))

    if ohe_cols is None and ord_cols is None and num_cols is None:
        raise Exception('There exists no list for pipeline')
    if ohe_cols is not None:
        ohe_transformer = Pipeline(steps=[('ohe', OneHotEncoder(drop='first',
                                                                sparse_output=False,
                                                                handle_unknown='ignore').
                                           set_output(transform="pandas"))])
        transformers['ohe'] = ('ohe', ohe_transformer, ohe_cols)
    if ord_cols is not None:
        ord_transformer = Pipeline(steps=[('ord', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                                 unknown_value=np.nan,
                                                                 encoded_missing_value=np.nan).
                                           set_output(transform="pandas"))])
        transformers['ord'] = ('ord', ord_transformer, ord_cols)
    if num_cols is not None:
        if is_std_scaler:
            scaler = StandardScaler().set_output(transform="pandas")
        else:
            scaler = MinMaxScaler().set_output(transform="pandas")
        num_transformer = Pipeline(steps=[('num', scaler)]).set_output(transform="pandas")
        transformers['num'] = ('num', num_transformer, num_cols)
    b_first = True
    for name, transformer in transformers.items():
        if transformer is not None:
            if b_first is True:
                transformers = ColumnTransformer(transformers=[transformer],
                                                 remainder='passthrough', n_jobs=-1,
                                                 verbose_feature_names_out=False)
                b_first = False
            else:
                transformers.transformers.append(transformer)
    pipeline = Pipeline(steps=[('preprocessor', transformers)]).set_output(transform="pandas")
    return pipeline


def print_means(df: pd.DataFrame):
    for col in df.columns:
        print(f'{col}: mean-> {df[col].mean()}, std-> {df[col].std()}')


def fit_models(test_x: pd.DataFrame, test_y: pd.DataFrame, models: list):
    class_report = list()
    conf_matrix = list()
    accuracies = list()
    for model in models:
        y_pred = model.predict(test_x)
        class_report.append(classification_report(test_y, y_pred))
        conf_matrix.append(confusion_matrix(test_y, y_pred))
        accuracies.append(accuracy_score(test_y, y_pred))
    return class_report, conf_matrix, accuracies


def print_reports(model_names, class_report, conf_matrix, accuracies):
    for i, model in enumerate(model_names):
        print('*' * 30)
        print(model)
        print('*' * 30)
        print(class_report[i])
        print('*' * 5)
        print(conf_matrix[i])
        print('*' * 5)
        print(accuracies[i])


def plot_heatmap(df: pd.DataFrame):
    """
    Plots the correlation matrix and shows correlation between features.
    Args:
        df: dataset whose correlation matrix will be shown
    Returns:

    """
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    heatmap = sns.heatmap(corr, mask=mask, vmax=1, center=0,
                          square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)


def plot_bar(df, title, limit=0):
    if limit == 0:
        limit = len(df.columns.tolist())
    sorted_mean = df.mean().abs().sort_values(ascending=False).head(limit)
    sorted_std = df.std().loc[sorted_mean.index]
    fig, ax = plt.subplots()
    ax.barh(sorted_mean.index, sorted_mean.values, yerr=sorted_std.values, align='center', alpha=0.5,
            capsize=10)
    plt.xlabel('MSE')
    plt.ylabel('Imputation Strategies')
    plt.title(title)
    plt.show()


def plot_mean(df, title, limit=0):
    if limit == 0:
        limit = len(df.columns.tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    means = (-df.mean()).sort_values(ascending=False).head(limit)
    errors = df.std().loc[means.index]
    means.plot.barh(xerr=errors, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("MSE")
    ax.set_yticks(np.arange(means.shape[0]))
    ax.set_yticklabels(means.index)
    plt.tight_layout(pad=1)
    plt.show()


def plot_means_two_dataset(df_list, title):
    means = []
    stds = []
    for df in df_list:
        means.append(-df.mean())
        stds.append(df.std())
    df = {'Mean': pd.concat(means), 'Std': pd.concat(stds)}
    df = pd.DataFrame(df)
    df = df.sort_values(by='Mean', ascending=False)
    plt.barh(df.index, df['Mean'])
    plt.errorbar(df['Mean'], df.index, xerr=df['Std'], fmt='o')
    plt.ylabel('Imputations')
    plt.xlabel('MSE')
    plt.title(title)
    plt.show()


def plot_frequency_causal_list(l_causal):
    # flatten
    flat_list = np.array([item for sublist in l_causal.values() for sublist2 in sublist for item in sublist2])
    categories, counts = np.unique(flat_list, return_counts=True)

    # Sort categories by frequency
    sorted_idx = np.argsort(-counts)
    categories = categories[sorted_idx]
    counts = counts[sorted_idx]
    

    # create a bar chart
    fig, ax = plt.subplots(figsize=(8, 6)) # set figure size
    ax.barh(categories, counts)

    # add frequency values to the plot
    for i, v in enumerate(counts):
        ax.text(v + 0.2, i, str(v), color='blue')

    # set plot title and axis labels
    ax.set_title('Frequency of Features')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Features')

    # display the plot
    plt.show()