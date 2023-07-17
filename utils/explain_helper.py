import itertools
from multiprocessing import Pool
from os import cpu_count

import eli5
import numpy as np
import pandas as pd
import shap
from IPython.utils import io
from dice_ml.utils.exception import UserConfigValidationException
from eli5.sklearn import PermutationImportance
from . import helper
seed = 42

def make_groups(df, pipe, num_cols, ord_cols=None):
    columns = df.columns.tolist()
    num_cols = [col for col in num_cols if col in columns]
    ord_cols = [col for col in ord_cols if col in columns]

    group_names = []
    for (name, enc, feat) in pipe.named_steps['preprocessor'].transformers_:
        if not isinstance(feat[0], int):
            group_names = group_names + feat
        if name == 'ohe' and 'ohe' in enc.named_steps:
            encoder = enc.named_steps['ohe']
    if encoder is None:
        raise Exception("There exist no ohe columns")

    feat_start_pos = list()
    feat_start_pos.append(0)
    counter = 0
    for cat in encoder.categories_:
        counter += len(cat) - 1
        feat_start_pos.append(counter)

    out_names = encoder.get_feature_names_out().tolist()
    group_feat = []
    groups = []
    for i in range(len(feat_start_pos) - 1):
        group_feat.append(out_names[feat_start_pos[i]: feat_start_pos[i + 1]])
    for feat in group_feat:
        grp = []
        for ind in feat:
            grp.append(df.columns.get_loc(ind))
        groups.append(grp)
    for ind in num_cols:
        groups.append([df.columns.get_loc(ind)])
    if ord_cols is not None:
        for ind in ord_cols:
            groups.append([df.columns.get_loc(ind)])

    return sorted(groups), group_names


def print_pred(df_x, df_y, model, ind):
    x_val = np.array(df_x.iloc[ind]).reshape(1, -1)
    pred = model.predict(x_val)
    actual = df_y.iloc[ind]
    print(f'{ind}. element with actual class {actual} was predicted {pred[0]}')


def show_importance(x_values: pd.DataFrame, y_values: pd.DataFrame, model_n):
    feature_list = x_values.columns.tolist()
    feature_perf = PermutationImportance(model_n, random_state=42).fit(x_values, y_values)
    return eli5.show_weights(feature_perf, feature_names=feature_list)


def plot_element(explainer, shap_v, data, ind, class_n=0):
    return shap.force_plot(
        explainer.expected_value[class_n],
        shap_v[class_n][ind, :],
        data.iloc[ind, :],
        link='logit')


def get_all_perm(columns):
    perm = list()
    for i in range(1, len(columns) + 1):
        for subset in itertools.combinations(columns, i):
            perm.append(list(subset))
    return perm


def is_necessary(instance, dice_gen, features_to_vary):
    """
    it shows if cols are sufficient for model to predict.
    :param instance: pd.Dataframe
    :param dice_gen: dice object
    :param features_to_vary: columns to test if they are sufficient for model to predict
    :return: True, if it satisfies the sufficient feature, otherwise false:
    """
    total_CFs = 1
    try:
        dice_gen.generate_counterfactuals(instance, total_CFs=total_CFs, 
                                          features_to_vary=features_to_vary, verbose=False)
        return True
    except UserConfigValidationException:
        return False
    except Exception as e:
        print(f'Error with {e}')


def is_sufficiency(instance, columns, dice_gen, features_to_constant):
    """
    it shows if cols are necessary for model to predict.
    :param columns:
    :param instance: pd.Dataframe
    :param dice_gen: dice object
    :param features_to_constant: columns to test if they are sufficient for model to predict
    :return: True, if it satisfies the sufficient feature, otherwise false:
    """
    total_CFs = 1
    features_to_vary = [col for col in columns if col not in features_to_constant]
    try:
        dice_gen.generate_counterfactuals(instance, total_CFs=total_CFs, features_to_vary=features_to_vary,
                                          verbose=False)
        return False
    except UserConfigValidationException:
        return True
    except Exception as e:
        raise Exception(f'Error with {e}')


def subset_excluding_superset(lists, subset_length, exclude_lists):
    subsets = []
    for item in itertools.combinations(lists, subset_length):
        subsets.append(set(item))
    if not exclude_lists:
        return [list(subset) for subset in subsets], []
    # remove subsets that are subsets of elements in lists
    subset_c = [list(subset) for subset in subsets if not any(subset.issuperset(elem) for elem in exclude_lists)]
    other = [list(subset) for subset in subsets if any(subset.issuperset(elem) for elem in exclude_lists)]
    return subset_c, other


def get_suf_necessities(instance, columns, dice_gen):
    to_changes = []
    not_changes = []
    min_changes = []
    for i in range(1, len(columns)+1):
        subsets, already_changed = subset_excluding_superset(columns, i, to_changes)
        to_changes.extend(already_changed)
        for cols in subsets:
            b_necessity = is_necessary(instance, dice_gen, cols)
            if b_necessity:
                to_changes.append(cols)
                min_changes.append(cols)
            else:
                not_changes.append(cols)
    return {'necessity': to_changes,
            'min_necessity': min_changes,
            'sufficiency': not_changes,
            'suf_necessity': find_suf_necessity(columns, to_changes, not_changes)}


def find_suf_necessity(columns, to_changes, not_changes):
    suf_necessity = []
    for cols in to_changes:
        not_vary = [col for col in columns if col not in cols]
        if not not_vary:
            suf_necessity.append(columns)
        if not_vary in not_changes:
            suf_necessity.append(cols)
    return suf_necessity


def parallel_helper(args):
    i, df, exp_dice = args
    if i % 10 == 0:
        print(i)
    instance = df.iloc[i:i + 1]
    ind = instance.index[0]
    with io.capture_output() as captured:
        return ind, get_suf_necessities(instance, df.columns.tolist(), exp_dice)


def find_all_causality(df, exp_dice):
    all_causalities = dict()
    args = [(i, df, exp_dice) for i in range(len(df)) ]
    with Pool(processes=cpu_count()) as pool:
        for ind, suf_necessities in pool.map(parallel_helper, args):
            all_causalities[ind] = suf_necessities
    return all_causalities


def plot_element(explainer, shap_v, data, ind, class_n=0):
    return shap.force_plot(
        explainer.expected_value[class_n],
        shap_v[class_n][ind, :],
        data.iloc[ind, :],
        link='logit')


def find_unique_lists(lists):
    unique_list = [lst for i, lst in enumerate(lists) if not any(set(lst).issuperset(lst2) 
                   and lst != lst2 for j, lst2 in enumerate(lists[:i]+lists[i+1:]))]
    return unique_list

def col_feat_map(df):
    num_cols = helper.get_numerical_columns(df)
    cat_cols = helper.get_categorical_columns(df)
    feat_map = dict()
    feat_map['num'] = {}
    feat_map['cat'] = {}
    
    if num_cols:
        for col in num_cols:
            min = df[col].min()
            max = df[col].max()
            feat_map['num'][col] = {'min' : min,
                                    'max' : max 
                                    }
    if cat_cols:
        for col in cat_cols:
            feat_map['cat'][col] = list(df[col].unique())
    
    return feat_map

def add_random_values(row, feat_map_dict, feat_to_vary):
    npg = np.random.RandomState(seed)
    row_cpy = row.copy()
    cat_map = feat_map_dict['cat']
    num_map = feat_map_dict['num']
    if cat_map:
        cat_cols = list(cat_map.keys())
        for col in feat_to_vary:
            if col in cat_cols:
                row_cpy[col] = npg.choice(cat_map[col])
    if num_map:
        num_cols = list(num_map.keys())
        for col in feat_to_vary:
            if col in num_cols:
                row_cpy[col] = npg.uniform(num_map[col]['min'], num_map[col]['max'])
    return row_cpy


def add_randomness(df_train, df_test, causal_dict):
    non_missing = pd.DataFrame()
    random_df =  pd.DataFrame()
    other_random =  pd.DataFrame()
    cols = df_test.columns.tolist()
    feat_map = col_feat_map(pd.concat([df_train, df_test], ignore_index=True))
    for i in range(len(df_test.index)):
        row = df_test.iloc[i:i+1]
        ind = row.index[0]
        for features in causal_dict[ind]:
            non_missing = pd.concat([non_missing, row], axis = 0, ignore_index=True)
            new_row = add_random_values(row, feat_map, features)
            random_df = pd.concat([random_df, new_row], axis = 0, ignore_index=True)
            non_feat = [col for col in cols if col not in features]
            row_other = add_random_values(row, feat_map, non_feat)
            other_random = pd.concat([other_random, row_other], axis=0, ignore_index=True)
    assert len(non_missing) == len(random_df)
    return non_missing, random_df, other_random

