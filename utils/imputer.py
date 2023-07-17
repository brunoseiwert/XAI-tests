import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from . import helper
from sklearn import set_config
import copy
from miceforest import ImputationKernel
from miceforest import mean_match_shap

set_config(transform_output="pandas")
seed = 38
random = np.random.RandomState(seed)


def add_missing_values(df: pd.DataFrame, missing_rate: float = 10):
    """
    This function adds missing values to a pandas DataFrame by selecting random
    indices and features and replacing their values with NaN.

    Parameters:
    df :pd.DataFrame
        The original DataFrame
    missing_rate : float
        The percentage of missing values to add to the DataFrame (default: 10)

    Returns:
    pd.DataFrame: The DataFrame with added missing values.
    """
    df_missing = df.copy()
    n_samples, n_features = df_missing.shape
    n_missing_samples = int(n_samples * missing_rate / 100)

    # Select random indices and features for missing values
    missing_features = random.randint(0, n_features, n_missing_samples)
    missing_index = random.randint(0, n_samples, n_missing_samples)

    # Replace the values at the selected indices and features with NaN
    for ind, col in zip(missing_index, missing_features):
        df_missing.iat[ind, col] = np.nan

    return df_missing


def add_missing_to_feature(df, missing_rate, feature):
    """
    Add missing values to the specified feature in a dataframe.

    Parameters:
    df :pd.DataFrame
        The original DataFrame
    missing_rate : float
        The percentage of missing values to add to the DataFrame
    feature: str
        The name of the feature to add missing values to.

    Returns:
    pd.DataFrame: The DataFrame with added missing values.
    """

    df_missing = df.copy()
    n_samples, _ = df_missing.shape
    n_missing_samples = int(n_samples * missing_rate / 100)
    # Select random indices for given feature
    missing_index = random.randint(0, n_samples, n_missing_samples)

    df_missing.iloc[missing_index][feature] = np.nan
    return df_missing


def get_ord_encoder(cat_cols):
    pipe = Pipeline(steps=[('ord', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                  unknown_value=np.nan,
                                                  encoded_missing_value=np.nan))]). \
        set_output(transform="pandas")
    return ColumnTransformer(transformers=[('ord_col', pipe, cat_cols)],
                             remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)


def get_common_impute(num_cols, cat_cols):
    simple_num = Pipeline(steps=[('mean', SimpleImputer())]).set_output(transform="pandas")
    simple_col = Pipeline(steps=[('most_freq', SimpleImputer(strategy='most_frequent'))]). \
        set_output(transform="pandas")
    return ColumnTransformer(transformers=[('num_col', simple_num, num_cols), ('cat_col', simple_col, cat_cols)],
                             remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)


def get_rand_forest_impute(num_cols, cat_cols):
    cat_rf = RandomForestClassifier(random_state=seed)
    num_rf = RandomForestRegressor(random_state=seed)
    return ColumnTransformer(transformers=[
        ('num', IterativeImputer(estimator=num_rf, initial_strategy='mean', random_state=seed), num_cols),
        ('cat', IterativeImputer(estimator=cat_rf, initial_strategy='most_frequent', random_state=seed), cat_cols)],
        remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)


def get_bayesian_impute():
    return Pipeline(steps=[('bayesian', IterativeImputer(initial_strategy='most_frequent', random_state=seed))]). \
        set_output(transform="pandas")


def get_scaler(num_cols, b_standard: bool = False):
    if b_standard:
        scaler = StandardScaler().set_output(transform="pandas")
    else:
        scaler = MinMaxScaler().set_output(transform="pandas")
    num_transformer = Pipeline(steps=[('scaler', scaler)]).set_output(transform="pandas")
    return ColumnTransformer(transformers=[('num_cols', num_transformer, num_cols)],
                             remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)


def get_knn_impute(num_cols, cat_cols, n_neighbors=5):
    num_knn = Pipeline(steps=[('num_knn', KNNImputer(weights='distance', n_neighbors=n_neighbors))]). \
        set_output(transform="pandas")
    cat_knn = Pipeline(steps=[('num_knn', KNNImputer(weights='uniform', n_neighbors=n_neighbors))]). \
        set_output(transform="pandas")
    return ColumnTransformer(transformers=[('num_col', num_knn, num_cols), ('cat_col', cat_knn, cat_cols)],
                             remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)


def create_imputation_pipeline(num_cols: list, cat_cols: list):
    ord_trans = get_ord_encoder(cat_cols)
    common_imp = get_common_impute(num_cols, cat_cols)
    knn_imp = get_knn_impute(num_cols, cat_cols)
    rf_imp = get_rand_forest_impute(num_cols, cat_cols)
    bayesian_imp = get_bayesian_impute()

    imputations = [common_imp, knn_imp, rf_imp, bayesian_imp]
    pipelines = dict()
    pipe_names = ['mean/most_freq', 'knn', 'random_forest', 'bayesian_ridge']

    for imp, p_name in zip(imputations, pipe_names):
        pipelines[p_name] = Pipeline(steps=[('enc', ord_trans), ('imp', imp)]).set_output(transform="pandas")

    return pipelines


def get_impute_pipe(num_cols, cat_cols, imp_name):
    steps = []
    if cat_cols is not None:
        ord_encoder = get_ord_encoder(cat_cols)
        steps.append(('enc', ord_encoder))
    imp_steps = {'mean/most_freq': get_common_impute(num_cols, cat_cols),
                 'knn': get_knn_impute(num_cols, cat_cols),
                 'random_forest': get_rand_forest_impute(num_cols, cat_cols),
                 'bayesian_ridge': get_bayesian_impute()
                 }
    if imp_name in imp_steps.keys():
        steps.append(('imp', imp_steps[imp_name]))
    else:
        raise Exception('Imputation name does not conform to available imputation strategies.')
    return Pipeline(steps=steps).set_output(transform="pandas")


def whole_impute_inverse(df_x, pipe, num_cols, cat_cols: list = None, b_inverse: bool = True):
    transformed = pipe.fit_transform(df_x)
    if b_inverse and cat_cols is not None:
        print(type(transformed))
        encoder = pipe.named_steps['enc'].transformers_[0][1].named_steps['ord']
        transformed = inverse_ord(transformed, encoder, cat_cols, num_cols)

    return transformed


def all_impute(df_x, df_y, models, num_cols, cat_cols, ord_cols=None, scoring='min_squared_error',
               b_feat_elim: bool = False, b_inverse: bool = True, oversampling: bool = True,
               b_single: bool = False):
    pipes = create_imputation_pipeline(num_cols, cat_cols)
    imputation_scores = pd.DataFrame()

    for p_name, pipe in pipes.items():
        print(f'{p_name} is working...')
        print('*' * 30)
        if b_inverse:
            p_name += ' + inverse'
        if b_single:
            p_name = 'single + ' + p_name
            transformed = single_impute_inverse(df_x, pipe, num_cols, cat_cols, b_inverse)
        else:
            transformed = whole_impute_inverse(df_x, pipe, num_cols, cat_cols, b_inverse)

        scores = evaluate_imputation(transformed, df_y, num_cols, cat_cols, models, ord_cols,
                                     b_feat_elim, scoring, oversampling)
        scores = change_column_name(scores, p_name)
        imputation_scores = pd.concat([imputation_scores, scores], axis=1)

    return imputation_scores


def split_na_rows(df: pd.DataFrame):
    non_missing = df.dropna()
    missing = df[df.isna().any(axis=1)]
    assert df.shape[0] == missing.shape[0] + non_missing.shape[0]
    return non_missing, missing


def single_impute_inverse(df_x: pd.DataFrame, pipe, num_cols, cat_cols: list = None,
                          b_inverse: bool = True):
    df = df_x.copy()
    df = df[cat_cols + num_cols]
    df_non_missed, df_missed = split_na_rows(df)
    length = len(df_non_missed.index)
    result = df_missed.copy()
    if not b_inverse:
        result = pd.DataFrame()
    for i in range(len(df_missed.index)):
        row = df_missed.iloc[i:i+1]
        ind = row.index[0]
        df_new = pd.concat([df_non_missed, row], axis=0, ignore_index=True)
        df_new = pipe.fit_transform(df_new)
        if b_inverse and cat_cols is not None:
            encoder = pipe.named_steps['enc'].transformers_[0][1].named_steps['ord']
            transformed = inverse_ord(df_new, encoder, cat_cols, num_cols)
            result.loc[ind] = transformed.loc[length]
        else:
            result = pd.concat([result, df_new.iloc[length]], axis=0, ignore_index=True)
    result = pd.concat([result, df_non_missed])

    return result


def inverse_ord(df, encoder, cat_cols, num_cols):
    transformed_cat = pd.DataFrame(encoder.inverse_transform(df[cat_cols]),
                                   columns=cat_cols)
    if num_cols is not None:
        transformed_cat = pd.concat([transformed_cat, df[num_cols].reset_index(drop=True)], axis=1)
    return transformed_cat



def evaluate_imputation(df_x, df_y, num_cols, cat_cols, models, ord_cols=None, b_feat_elim=True,
                        scoring='accuracy', oversampling=True):
    p_name = ''

    nums = copy.deepcopy(num_cols)
    cats = copy.deepcopy(cat_cols)
    ords = copy.deepcopy(ord_cols)

    transformed = df_x.copy()

    if b_feat_elim:
        transformed, nums, cats = helper.eliminate_feats(transformed, df_y, nums, cats)
        p_name += 'feat_elim'

    if ord_cols is not None:
        ords = [col for col in ords if col in cats]
        if len(ord_cols) == 0:
            ords = None
        else:
            cats = [col for col in cats if col not in ords]

    pipeline = helper.create_pipeline(cats, ords, nums, False)
    pipeline.fit(transformed)
    transformed = pipeline.transform(transformed)

    scores = helper.evaluate_all_models(transformed, df_y, models, scoring, oversampling)
    scores = change_column_name(scores, p_name)
    return scores


def change_column_name(df, prev):
    df.columns = [prev + ' + ' + col for col in df.columns.tolist()]
    return df


def to_category(df: pd.DataFrame, cat_cols):
    if cat_cols is None:
        return df
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df


def get_mice(df: pd.DataFrame, imputed_columns: list = None):
    df_cpy = df.copy()

    kernel = ImputationKernel(
        data=df_cpy,
        variable_schema=imputed_columns,
        mean_match_scheme=mean_match_shap,  # use shap values for KNN slower but better quality
        save_all_iterations=True,
        random_state=seed,
    )
    kernel.mice(5)  # 5 mal iterations
    return kernel


def single_impute_mice(df_x, cat_cols):
    non_missed_df, missed_df = split_na_rows(df_x)
    result = missed_df.copy()

    for i in range(len(missed_df.index)):
        row = missed_df.iloc[i:i+1]
        ind = row.index[0]
        df_new = pd.concat([non_missed_df, row], axis=0, ignore_index=True)
        to_filled_cols = [col for col in df_new.columns if df_new[col].isna().any() and col != 'Index']
        df_new = to_category(df_new, cat_cols)

        kernel = get_mice(df_new, to_filled_cols)
        df_new = kernel.complete_data(dataset=0)
        result.loc[ind] = df_new.iloc[-1]
    result = pd.concat([result, non_missed_df])
    return result


def whole_impute_mice(df_x, cat_cols):
    df = df_x.copy()
    if cat_cols is not None:
        df = to_category(df, cat_cols)
    kernel = get_mice(df)
    completed = kernel.complete_data(dataset=0)
    return completed


def all_mice(df_x, df_y, models, num_cols, cat_cols, ord_cols=None, scoring='min_squared_error',
             b_feat_elim: bool = False, oversampling: bool = True,
             b_single: bool = False):
    p_name = 'mice'
    if b_single:
        transformed = single_impute_mice(df_x, cat_cols)
        p_name += ' + single'
    else:
        transformed = whole_impute_mice(df_x, cat_cols)
        p_name += ' + whole'
    scores = evaluate_imputation(transformed, df_y, num_cols, cat_cols, models, ord_cols,
                                 b_feat_elim, scoring, oversampling)
    scores = change_column_name(scores, p_name)
    return scores


def impute_feat(df, df_y, num_cols, cat_cols, imp_name, model, ord_cols, miss_rate=10, feat_elim=True,
                scoring='neg_mean_squared_error', oversampling=True):
    df_cpy = df.copy()
    rf_pipe = get_impute_pipe(num_cols, cat_cols, imp_name)
    feat_scores = pd.DataFrame()
    for col in df.columns.tolist():
        df_missed = add_missing_to_feature(df_cpy, miss_rate, col)
        transformed = rf_pipe.fit_transform(df_missed)
        if cat_cols is not None:
            encoder = rf_pipe.named_steps['enc'].transformers_[0][1].named_steps['ord']
            transformed = inverse_ord(transformed, encoder, cat_cols, num_cols)
        feat_scores[col] = evaluate_imputation(transformed, df_y, num_cols,
                                               cat_cols, model, ord_cols, feat_elim, scoring, oversampling)
    return feat_scores
