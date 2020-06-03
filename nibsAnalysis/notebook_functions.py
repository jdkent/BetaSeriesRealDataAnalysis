"""
functions to help with keeping the notebook uncluttered
(and allow testing of the functions)
"""
import re
from multiprocessing.pool import Pool
from functools import partial
import random

import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from nilearn.plotting import plot_connectome
from scipy import stats
from joblib import Parallel, delayed

from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from bct.algorithms.modularity import community_louvain, modularity_louvain_und_sign
from bct.algorithms.clustering import clustering_coef_wu_sign
from bct.algorithms.centrality import participation_coef_sign


def _read_adj_matrix(file):
    """read the adjacency matrix file
    """
    return pd.read_csv(file, index_col=0, sep='\t')


def _compare_networks(row):
    """Identify whether the correlation in this row
    represents a "within" network correlation or
    a "between" network correlation
    
    Parameters
    ----------
    row : pandas.Series
    
    Returns
    -------
    str
    """
    if row['source_network'] == row['target_network']:
        return 'within'
    else:
        return 'between'

def _fishers_r_to_z(r):
    return np.arctanh(r)


def _fishers_z_to_r(z):
    """
    Parameters
    ----------
    z : float or numpy.array(like)

    Returns
    -------
    float or numpy.array(like)
    """
    return np.tanh(z)


def _adj_to_edge(df):
    """convert adjacency matrix to edge list
    """
    return nx.to_pandas_edgelist(nx.from_pandas_adjacency(df))


def _identify_within_between(edge_df):
    """
    specify whether an ROI-ROI pair is
    within a network (e.g., Vis-1__Vis-2)
    or between networks (e.g., Vis-1__Con-1)
    """
    
    edge_df.dropna(inplace=True) # drop self connections
    # isolate the networks from the longform schaefer names
    edge_df['source_network'] = edge_df['source'].str.split('-').apply(lambda x: x[1])
    edge_df['target_network'] = edge_df['target'].str.split('-').apply(lambda x: x[1])
    # specify whether the network-network correlation is within or between
    edge_df['network_connection'] = edge_df[['source_network', 'target_network']].apply(_compare_networks, axis=1)

    return edge_df


def _condense_within_between(proc_edge_df):
    """Give one measure for within and between"""
    info_df = proc_edge_df.groupby(['network_connection', 'source_network']).describe().T.loc[('weight', 'mean'), :].to_frame()
    info_df.columns = info_df.columns.droplevel()
    info_df.reset_index(inplace=True)

    return info_df


def summarize_network_correlations(file, participant_id, verbose=False):
    """
    Parameters
    ----------
    file : str
        filename of symmetric adjacency matrix
    participant_id : participant_id
        participant_id identifier

    Returns
    -------
    info_df : pandas.DataFrame
        dataframe with a within and between network measure for each network
    """
    adj_z_df = _read_adj_matrix(file)
    adj_r_df = _fishers_z_to_r(adj_z_df)
    
    edge_df = _adj_to_edge(adj_r_df)

    proc_edge_df = _identify_within_between(edge_df)

    info_df = _condense_within_between(proc_edge_df)

    info_df['participant_id'] = [participant_id] * info_df.shape[0]
    
    if verbose:
        print(f"finished {participant_id}")

    return info_df


def _subtract_matrices(file1, file2):
    """Take the difference between two adjacency matrices
    """
    df1 = _read_adj_matrix(file1)
    df2 = _read_adj_matrix(file2)
    diff_df = df1 - df2
    diff_edge_df = _adj_to_edge(diff_df)
    diff_edge_df.dropna(inplace=True)

    return diff_edge_df


def _proc_diff_df(edge_df, participant_id, verbose=False):
    """slightly redundant with summarize_network
    """
    proc_edge_df = _identify_within_between(edge_df)

    info_df = _condense_within_between(proc_edge_df)

    info_df['participant_id'] = [participant_id] * info_df.shape[0]

    info_df.reset_index(inplace=True)

    if verbose:
        print(f"finished {participant_id}")
    
    return info_df


def calc_diff_matrices(file1, file2, participant_id):
    """calculate the average within/between network correlation differences
    between two adjacency matrices
    """  
    edge_df = _subtract_matrices(file1, file2)
    
    out_df = _proc_diff_df(edge_df, participant_id)

    # translate the difference back to a Pearson's R.
    out_df['mean_r'] = _fishers_z_to_r(out_df['mean'])
    
    return out_df


def z_score_cutoff(arr, thresh):
    """give correlations very close to 1 a more reasonable z-score
    0.99 r == 2.647 z
    (this is the max z score I would be interested in,
     anything above does not explain meaningful differences)
    """
    return arr.clip(-thresh, thresh)


def _combine_adj_matrices_wide(dfs, names=None):
    """
    take a list of adjacency matrices
    and take all unique ROI-ROI pairs
    to create a unique column where
    there are two underscores separating
    the ROI names (e.g., Vis-1__Vis-2)
    """
    if names is None:
        names = dfs[0].columns

    upper_idxs = np.triu_indices(len(names), k=1)
    new_colnames = ['__'.join([names[i], names[j]]) for i, j in zip(*upper_idxs)]
    wide_df = pd.DataFrame(np.array([df.loc[names, names].values[upper_idxs] for df in dfs]), columns=new_colnames)

    return wide_df


def bind_matrices(objs, label, names=None):
    """combine all adjacency matrices to a wide format where
    each column represents a unique roi-roi pair and each row represents
    an observation from a participant_id.
    """

    dfs = [_read_adj_matrix(obj.path) for obj in objs]
    participant_ids = [obj.entities['subject'] for obj in objs]
    wide_df = _combine_adj_matrices_wide(dfs, names)
    wide_df['participant_id'] = participant_ids
    wide_df['task'] = [label] * len(participant_ids)

    return wide_df

def permute_condition_orig(df):
    """
    inputs
    ------
    df: pandas.Dataframe
        dataframe whose rows represent a participant duing a particular condition
 
    outputs
    -------
    df_copy: pandas.Dataframe
        copy of original dataframe with the correlation permuted
        to be either in one condition or another (e.g., switch or repeat).
    """
    df_copy = df.copy()
    participants = df['participant_id'].unique()
    for participant in participants:
        participant_rows = df['participant_id'] == participant
        column_selection = (df.columns != 'participant_id') & (df.columns != 'task')
        df_copy.loc[participant_rows, column_selection] = np.random.permutation(df.loc[participant_rows,column_selection])

    return df_copy


def permute_condition_one_sample(df, rng):
    """
    inputs
    ------
    df: pandas.Dataframe
        dataframe whose rows represent a participant's correlation difference between conditions
    rng: np.random.RandomState
        random number generator that contains reproducible state
     
    
    outputs
    -------
    df_copy: pandas.Dataframe
        copy of original dataframe with the correlation difference permuted
        to be either positive or negative (e.g., multiplied by either +1 or -1).
    """
 
    column_selector = (df.columns != 'participant_id') & (df.columns != 'task')
    correlation_values = df.loc[:, column_selector].values
    permutation = np.random.choice([-1, 1], size=correlation_values.shape)
    permutation_df = df.loc[:, column_selector] * permutation
    permutation_df[["participant_id", "task"]] = df[["participant_id", "task"]]
    
    return permutation_df


def permute_condition(df, rng):
    """
    inputs
    ------
    df: pandas.Dataframe
        dataframe whose rows represent a participant duing a particular condition
    rng: np.random.RandomState
        random number generator that contains reproducible state
     
    
    outputs
    -------
    df_copy: pandas.Dataframe
        copy of original dataframe with the correlation permuted
        to be either in one condition or another (e.g., switch or repeat).
    """
    df_copy = df.copy(deep=True)
    participants = df['participant_id'].unique()
    for participant in participants:
        participant_rows = df['participant_id'] == participant
        column_selection = (df.columns != 'participant_id') & (df.columns != 'task')
        df_copy.loc[participant_rows, column_selection] = rng.permutation(df.loc[participant_rows, column_selection])

    return df_copy


def _model_helper(wide_df, rng, use_python=False, permute=False, one_sample=False, alpha=0.05):
        model_df = model_corr_diff(wide_df, rng, use_python=use_python, permute=permute, one_sample=one_sample)
        return np.sum(model_df['p_value'] < alpha)


def count_positives_from_permutations_orig(objs1, objs2, trialtype1, trialtype2,
                                      permutations=2, alpha=0.05,
                                      nthreads=1, use_python=False):
    
    wide_df = pd.concat([bind_matrices(objs1, trialtype1),
                         bind_matrices(objs2, trialtype2)])
    
    # rng = np.random.RandomState(0)
    args = [(wide_df, np.random.RandomState(n)) for n in range(permutations)]
    # sig_pvalue_collector = []
    partial_rmod = partial(_model_helper, **{'use_python': use_python, 'permute': True, 'alpha': 0.05})
    sig_pvalue_collector = Parallel(n_jobs=32)(delayed(partial_rmod)(df,rng) for df, rng in args)
    # for _ in range(permutations):
    # for df, rng in args:
    #    n_sig_pvalues = partial_rmod(df, rng, use_python=use_python, permute=True)
        # model_df = model_corr_diff_mt(wide_df, nthreads, use_python=use_python, permute=True)
        # n_sig_pvalues = np.sum(model_df['p_value'] < alpha)
    #    print(n_sig_pvalues)
    #    sig_pvalue_collector.append(n_sig_pvalues)
    
    return sig_pvalue_collector


def count_positives_from_permutations(objs1=None, objs2=None, trialtype1=None, trialtype2=None,
                                      wide_df=None,
                                      permutations=2, alpha=0.05,
                                      nthreads=1, use_python=False,
                                      one_sample=False):
    
    if objs1:
        wide_df = pd.concat([bind_matrices(objs1, trialtype1),
                             bind_matrices(objs2, trialtype2)])
    elif wide_df is not None:
        pass
    else:
        raise ValueError("either objects or wide dataframe must be provided")
    
    args = [(wide_df, np.random.RandomState(n+10)) for n in range(permutations)]
    
    partial_rmod = partial(_model_helper, **{'use_python': use_python,
                                             'permute': True,
                                             'alpha': 0.05,
                                             'one_sample': one_sample})

    sig_pvalue_collector = Parallel(n_jobs=nthreads)(delayed(partial_rmod)(df,rng) for df, rng in args)

    return sig_pvalue_collector


def _run_model_python(df, col, nuisance_cols=None, one_sample=False):
    all_cols =['participant_id', 'task', col]
    if nuisance_cols:
        raise NotImplementedError("using nuisance columns is not implemented with python")
    filt_df = df[all_cols]
    filt_df = filt_df.rename({col: 'correlation'}, axis=1)
    # identify participants with nans and remove them
    inds = pd.isnull(filt_df).any(1).to_numpy().nonzero()[0]
    bad_participants = filt_df.iloc[inds, :]['participant_id'].unique()
    filt_good_df = filt_df[~filt_df['participant_id'].isin(bad_participants)]
    
    tasks = filt_good_df['task'].unique()
    collector_dict = {'source_target': None, 'p_value': None, 'estimate': None}
  
    filt_wide_df = filt_good_df.pivot(index='participant_id', columns='task', values='correlation')
    
    # compare to zero
    if one_sample:
        t, p = stats.ttest_1samp(filt_wide_df[tasks[0]], 0)
    else:
        t, p = stats.ttest_rel(filt_wide_df[tasks[0]], filt_wide_df[tasks[1]])
    
    collector_dict['source_target'] = col
    collector_dict['p_value'] = p
    collector_dict['estimate'] = t
    
    return collector_dict

STATS = importr('stats')
BASE = importr('base')

    
def _run_model(df, col, nuisance_cols=None, one_sample=False):
    if one_sample:
        raise NotImplementedError("one sample ttests are not implemented")
    all_cols = ['participant_id', 'task', col]
    if nuisance_cols:
        all_cols.extend(nuisance_cols)
    filt_df = df[all_cols]
    filt_df = filt_df.rename({col: 'correlation'}, axis=1)
    collector_dict = {'source_target': None, 'p_value': None, 'estimate': None}
    with localconverter(robj.default_converter + pandas2ri.converter):
        r_df = robj.conversion.py2rpy(filt_df)
    formula = 'correlation ~ task'
    if nuisance_cols:
        nuisance_cols = [formula] + nuisance_cols
        formula = '+'.join(nuisance_cols)
    model = STATS.lm(formula=formula, data=r_df)
    summary = BASE.summary(model)
    res = summary.rx2('coefficients')
    res = np.asarray(res) # sometimes is a FloatMatrix
    if res.ndim == 2:
        p_val = res[1][3]
        estimate = res[1][0]
    else:
        print("WARNING, CHECK THE RESULTS")
        p_val = res[7] # manually check this
        estimate = res[1] # manually check this
    # print(col)
    # print(res)
    collector_dict['source_target'] = col
    collector_dict['p_value'] = p_val
    collector_dict['estimate'] = estimate
    return collector_dict


def model_corr_diff(wide_df, rng, nuisance_cols=None, use_python=False, permute=False, one_sample=False):
    """Only runs permutations"""
    cols = set(wide_df.columns)
    # I do not want to iterate over these columns
    non_cols = ["task", "participant_id", "nan_rois", "num_nan_rois"]
    model_cols = ['participant_id', 'task']
    if nuisance_cols:
        non_cols.extend(nuisance_cols)
        model_cols.extend(nuisance_cols)
    cols = list(cols - set(non_cols))
    
    if use_python:
        model_runner = _run_model_python
    else:
        model_runner = _run_model
    
    dict_collector = {'source_target': [], 'p_value': [], 'estimate': []}
    if permute and one_sample:
        use_wide_df = permute_condition_one_sample(wide_df, rng)
    elif permute:
        use_wide_df = wide_df[model_cols].copy()
        for col in cols:
            use_wide_df[col] = permute_condition(wide_df[model_cols + [col]], rng)[col]
    else:
        use_wide_df = wide_df
    for col in cols:
        result_dict = model_runner(use_wide_df[model_cols + [col]], col, nuisance_cols, one_sample)
        for k in ['source_target', 'p_value', 'estimate']:
            dict_collector[k].append(result_dict[k])
    
    return pd.DataFrame.from_dict(dict_collector)
        
    
def model_corr_diff_mt(wide_df, n_threads, nuisance_cols=None, use_python=False, permute=False, one_sample=False):
    """setup to run linear regression for every roi-roi pair
    """
    cols = set(wide_df.columns)
    # I do not want to iterate over these columns
    non_cols = ["task", "participant_id", "nan_rois", "num_nan_rois"]
    model_cols = ['participant_id', 'task']
    if nuisance_cols:
        non_cols.extend(nuisance_cols)
        model_cols.extend(nuisance_cols)
    cols = list(cols - set(non_cols))
    
    if permute:
        rng = np.random.RandomState(0)
        args = [(permute_condition(wide_df[model_cols + [col]], rng), col, nuisance_cols, one_sample) for col in cols]
    else:
        args = [(wide_df[model_cols + [col]], col, nuisance_cols, one_sample) for col in cols]
    if use_python:
        model_runner = _run_model_python
    else:
        model_runner = _run_model
    # run this in parallel to speed up computation
    with Pool(n_threads) as p:
        sep_dicts = p.starmap(model_runner, args)
    dict_collector = {
            k: [d.get(k) for d in sep_dicts]
            for k in set().union(*sep_dicts)}
    model_df = pd.DataFrame.from_dict(dict_collector)
    return model_df


def _flip_hemisphere_network(w):
    """schaefer roi names look like LH-ContA_SPL_1,
    this changes them to look like ContA-LH_SPL_1
    """
    comps = w.split('-')
    return '-'.join([comps[1], comps[0]]) + '_' + ''.join(comps[2:])


def _edge_to_adj(df, measure):
    df[["source", "target"]] = df["source_target"].str.split('__', expand=True)
    return nx.to_pandas_adjacency(nx.from_pandas_edgelist(df, edge_attr=measure), weight=measure)


def _sort_columns(df):
    df_rename = df.rename(_flip_hemisphere_network, axis=1).rename(_flip_hemisphere_network, axis=0)
    col_list = df_rename.columns.tolist()
    col_list.sort()

    return df_rename.loc[col_list, col_list]


def make_symmetric_df(df, measure):
    tmp_df = _edge_to_adj(df, measure)
    pretty_df = _sort_columns(tmp_df)
    
    return pretty_df


def _make_pretty_schaefer_heatmap(adj_df, **hm_kwargs):
    if adj_df.shape[0] != adj_df.shape[1]:
        raise ValueError("The dataframe is not square")

    # get the network assignments (assumed name is like ContA-LH_SPL_1)
    networks = adj_df.columns.str.split('-', n=1, expand=True).get_level_values(0)
    # at what indices do the networks change (e.g., go from ContA to ContB)
    # https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy
    network_change_idxs = np.where(np.roll(networks,1)!=networks)[0]
    # find the midpoint index in each network to place a label
    tmp_idx = np.append(network_change_idxs, adj_df.shape[0])
    midpoints = (tmp_idx[1:] + tmp_idx[:-1]) // 2
    
    # create figure axes to plot onto
    fig, ax = plt.subplots(figsize=(12, 10))
    # make the heatmap
    ax = sns.heatmap(adj_df, ax=ax, **hm_kwargs)
    # add horizontal (hlines) and vertical (vlines) to delimit networks
    ax.hlines(network_change_idxs, xmin=0, xmax=adj_df.shape[0])
    ax.vlines(network_change_idxs, ymin=0, ymax=adj_df.shape[1])
    
    # remove ticklines on the axes
    ax.tick_params(length=0)
    
    # add network labels to the y-axis
    ax.set_yticks(midpoints)
    ax.set_yticklabels(networks.unique(), fontdict={'fontsize': 'medium', 'fontweight': 'heavy'}, va="center")

    # add network labels to the x-axis
    ax.set_xticks(midpoints)
    ax.set_xticklabels(networks.unique(), fontdict={'fontsize': 'medium', 'fontweight': 'heavy'}, ha="center")
    
    return fig


def get_lower_matrix_values(adj_matrix):
    cols = len(adj_matrix.columns)
    
    return adj_matrix.values[np.tril_indices(cols, k=-1)]


def from_1d_to_2d_lower_matrix(values, cols):
    # put it back into a 2D symmetric array
    n_cols = len(cols)
    X = np.zeros((n_cols, n_cols))
    X[np.tril_indices(n_cols, k=-1)] = values
    X = X + X.T - np.diag(np.diag(X))
    
    return pd.DataFrame(data=X, columns=cols, index=cols)


def return_lower_matrix(adj_matrix):
    """zeros the upper matrix and returns the lower matrix"""
    n_cols = len(adj_matrix.columns)
    lower_tri = get_lower_matrix_values(adj_matrix)
    
    return from_1d_to_2d_lower_matrix(lower_tri, adj_matrix.columns)


def make_glass_brain(overlap_df, coords_df):
    coords = coords_df[["X", "Y", "Z"]].values
    coord_names = coords_df["Cluster ID"].values
    
    cmap_list = sns.xkcd_palette(["royal blue", "olive green", "bright yellow"])
    cmap=ListedColormap(cmap_list)
    
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), gridspec_kw={'width_ratios': [23/24, 1/24]})
    
    plot_connectome(overlap_df, axes=axes[0], figure=fig, edge_cmap=cmap, node_coords=coords,
                edge_vmin=0.9,
                edge_vmax=3,
                node_size=200,
                edge_threshold=0,
                colorbar=False)

    # make the color bar
    colorbar = mpl.colorbar.ColorbarBase(axes[1], cmap=cmap, orientation='vertical', boundaries=[0, 1, 2, 3])
    r = 3.0 
    colorbar.set_ticks([0.0 + r / 3 * (0.5 + i) for i in range(3)])
    colorbar.set_ticklabels(["LSA", "LSS", "Both"])
    colorbar.ax.tick_params(labelsize=12)

    # label the nodes with their number
    for ax in fig.axes[2:]:
        for name, node in zip(coord_names, ax.collections[0].get_offsets()):
            ax.annotate(name, node, ha='center', va='center', zorder=5000)
    
    return fig

    
def make_comparison_matrix(p_value_matrix1, matrix1_label, p_value_matrix2, matrix2_label, cmap_dict,
                           rois='schaefer'):
    
    # ENSURE CMAP_DICT IS IN CORRECT ORDER
    cols = p_value_matrix1.columns
    n_cols = len(cols)
    lower_tri_vals1 = get_lower_matrix_values(p_value_matrix1)
    lower_tri_vals2 = get_lower_matrix_values(p_value_matrix2)
    
    all_sig_vals1 = (lower_tri_vals1 < 0.05).astype(int)
    all_sig_vals2 = (lower_tri_vals2 < 0.05).astype(int)
    
    print("Number of Positives {lbl}: {pos} / {tot}".format(
        pos=all_sig_vals1.sum(), tot=all_sig_vals1.shape[0],
        lbl=matrix1_label))

    print("Number of Positives {lbl}: {pos} / {tot}".format(
        pos=all_sig_vals2.sum(), tot=all_sig_vals2.shape[0],
        lbl=matrix2_label))

    print("Number of overlapping positives: {pos} / {tot}".format(
        pos=(all_sig_vals1 * all_sig_vals2).sum(), tot=all_sig_vals1.shape[0]))
    
    # make each entry a 2 to make it unique relative to all_sig_vals1
    all_sig_vals2 = all_sig_vals2 * 2

    overlap_vals = all_sig_vals1 + all_sig_vals2

    overlap_df = from_1d_to_2d_lower_matrix(overlap_vals, cols)
    
    cmap = sns.xkcd_palette(["cyan", "royal blue", "olive green", "bright yellow"])
    
    if rois == 'schaefer':
        fig = _make_pretty_schaefer_heatmap(overlap_df,
                                            mask=np.triu(np.ones_like(overlap_df.values, dtype=np.bool)),
                                            cmap=cmap)
    elif rois == 'activation':
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(overlap_df,
                    mask=np.triu(np.ones_like(overlap_df.values, dtype=np.bool)),
                    cmap=cmap,vmin=0, vmax=3,
                    ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), fontdict={'fontsize': 'medium', 'fontweight': 'heavy'})
        ax.set_xticklabels(ax.get_xticklabels(), fontdict={'fontsize': 'medium', 'fontweight': 'heavy'})
    
    n = len(cmap_dict)
    colorbar = fig.axes[0].collections[0].colorbar
    r = 3.0 
    colorbar.set_ticks([0.0 + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(cmap_dict.keys()))
    colorbar.ax.tick_params(labelsize=20)
    
    return fig, overlap_df
    
    
def get_layout_objects(layout, trialtypes, **filters):
    object_collector = {}
    for trialtype in trialtypes:
        object_collector[trialtype] = layout.get(desc=trialtype, **filters)
    
    return object_collector
        

def compare_ppi_lss_lsa_sig(ppi_wide_df, lss_wide_df, lsa_wide_df, rois, nthreads=1, use_python=False):
    
    ppi_model_df = model_corr_diff_mt(ppi_wide_df, nthreads, use_python=use_python, one_sample=True)
    lss_model_df = model_corr_diff_mt(lss_wide_df, nthreads, use_python=use_python, one_sample=True)
    lsa_model_df = model_corr_diff_mt(lsa_wide_df, nthreads, use_python=use_python, one_sample=True)
    
    if rois == 'schaefer':
        ppi_pvalue_df = make_symmetric_df(ppi_model_df, "p_value")
        ppi_estimate_df = make_symmetric_df(ppi_model_df, "estimate")

        lss_pvalue_df = make_symmetric_df(lss_model_df, "p_value")
        lss_estimate_df = make_symmetric_df(lss_model_df, "estimate")

        lsa_pvalue_df = make_symmetric_df(lsa_model_df, "p_value")
        lsa_estimate_df = make_symmetric_df(lsa_model_df, "estimate")
    elif rois == 'activation':
        
        # .loc[columns, columns] ensures rois are listed in the same order
        ppi_pvalue_df = _edge_to_adj(ppi_model_df, "p_value")
        columns = ppi_pvalue_df.columns.sort_values()
        ppi_pvalue_df = ppi_pvalue_df.loc[columns, columns]
        ppi_estimate_df = _edge_to_adj(ppi_model_df, "estimate").loc[columns, columns]

        lss_pvalue_df = _edge_to_adj(lss_model_df, "p_value").loc[columns, columns]
        lss_estimate_df = _edge_to_adj(lss_model_df, "estimate").loc[columns, columns]
        
        lsa_pvalue_df = _edge_to_adj(lsa_model_df, "p_value").loc[columns, columns]
        lsa_estimate_df = _edge_to_adj(lsa_model_df, "estimate").loc[columns, columns]
    
    cmap_dict_lsa = {
        "Null": 0,
        "LSA": 1,
        "PPI": 2,
        "Both": 3
    }
    
    cmap_dict_lss = {
        "Null": 0,
        "LSS": 1,
        "PPI": 2,
        "Both": 3
    }

    fig_lsa, overlap_lsa_df = make_comparison_matrix(lsa_pvalue_df, "lsa", ppi_pvalue_df, "ppi", rois=rois, cmap_dict=cmap_dict_lsa)
    
    fig_lss, overlap_lss_df = make_comparison_matrix(lss_pvalue_df, "lss", ppi_pvalue_df, "ppi", rois=rois, cmap_dict=cmap_dict_lss)

    return fig_lsa, fig_lss, ppi_model_df, lss_model_df, lsa_model_df, overlap_lsa_df, overlap_lss_df


def compare_lss_lsa_sig(lss_objs1, lss_objs2, lsa_objs1, lsa_objs2,
                        trialtype1, trialtype2, rois, nthreads=1, use_python=False):

    lss_wide_df = pd.concat([bind_matrices(lss_objs1, trialtype1),
                             bind_matrices(lss_objs2, trialtype2)])

    lss_model_df = model_corr_diff_mt(lss_wide_df, nthreads, use_python=use_python)
    
    lsa_wide_df = pd.concat([bind_matrices(lsa_objs1, trialtype1),
                             bind_matrices(lsa_objs2, trialtype2)])
    
    lsa_model_df = model_corr_diff_mt(lsa_wide_df, nthreads, use_python=use_python)
    
    if rois == 'schaefer':
        lss_pvalue_df = make_symmetric_df(lss_model_df, "p_value")
        lss_estimate_df = make_symmetric_df(lss_model_df, "estimate")

        lsa_pvalue_df = make_symmetric_df(lsa_model_df, "p_value")
        lsa_estimate_df = make_symmetric_df(lsa_model_df, "estimate")
    elif rois == 'activation':
        columns = _read_adj_matrix(lss_objs1[0].path).columns
        # .loc[columns, columns] ensures rois are listed in order
        lss_pvalue_df = _edge_to_adj(lss_model_df, "p_value").loc[columns, columns]
        lss_estimate_df = _edge_to_adj(lss_model_df, "estimate").loc[columns, columns]
        
        lsa_pvalue_df = _edge_to_adj(lsa_model_df, "p_value").loc[columns, columns]
        lsa_estimate_df = _edge_to_adj(lsa_model_df, "estimate").loc[columns, columns]
    
    cmap_dict = {
        "Null": 0,
        "LSA": 1,
        "LSS": 2,
        "Both": 3
    }

    fig, overlap_df = make_comparison_matrix(lsa_pvalue_df, "lsa", lss_pvalue_df, "lss", rois=rois, cmap_dict=cmap_dict)
    
    return fig, lss_model_df, lsa_model_df, overlap_df


def _identify_nan_entries(adj_df):
    adj_arr = adj_df.values
    np.fill_diagonal(adj_arr, 0)

    rois = adj_df.columns
    # assuming nans are constant through a roi
    # also assuming the first roi is not nan
    nan_idxs = np.argwhere(np.isnan(adj_arr[0,:]))
    nan_rois = rois[nan_idxs]

    return nan_idxs, nan_rois, len(nan_rois)


def _run_graph_theory_measure(adj_df, graph_func, **graph_kwargs):
    nan_idxs, nan_rois, num_nan_rois = _identify_nan_entries(adj_df)
    adj_df = adj_df.drop(labels=nan_rois, axis=1).drop(labels=nan_rois, axis=0)
    adj_arr = adj_df.values

    np.fill_diagonal(adj_arr, 0)

    if 'ci' in graph_kwargs.keys() and nan_rois.any():
        graph_kwargs['ci'] = np.delete(graph_kwargs['ci'], nan_idxs)

    graph_output = graph_func(adj_arr, **graph_kwargs)

    return nan_rois, num_nan_rois, graph_output


def calc_modularity(file, participant_id, task):
    # read the file
    adj_df = _read_adj_matrix(file)
    
    # rename and sort the columns
    adj_df = _sort_columns(adj_df)
    
    adj_r_df = _fishers_z_to_r(adj_df)

    # pass the original community classification (based on schaefer)
    orig_ci = adj_r_df.columns.str.split('-', n=1, expand=True).get_level_values(0)

    # run modularity
    nan_rois, num_nan_rois, (ci, modularity) = _run_graph_theory_measure(
        adj_r_df, community_louvain, B='negative_asym', ci=orig_ci, gamma=1.295,
    )

    num_ci = len(np.unique(ci))

    result_dict = {
        'nan_rois': nan_rois,
        'num_nan_rois': num_nan_rois,
        'ci': ci,
        'num_ci': num_ci,
        'modularity': modularity,
        'participant_id': participant_id,
        'task': task,
    }

    return result_dict


def calc_clustering_coef(file, participant_id, task):
    # read the file
    adj_df = _read_adj_matrix(file)
    
    # rename and sort the columns
    adj_df = _sort_columns(adj_df)
    
    adj_r_df = _fishers_z_to_r(adj_df)

    nan_rois, num_nan_rois, cluster_coefs = _run_graph_theory_measure(
        adj_r_df, clustering_coef_wu_sign, coef_type='constantini'
    )

    # combine cluster coefs with their respective rois
    track_idx = 0
    cluster_coef_dict = {}
    for roi in adj_df.columns:
        if roi in list(nan_rois):
            cluster_coef_dict[roi] = np.nan
        else:
            cluster_coef_dict[roi] = cluster_coefs[track_idx]
            track_idx += 1

    cluster_coef_dict['nan_rois'] = nan_rois
    cluster_coef_dict['num_nan_rois'] = num_nan_rois
    cluster_coef_dict['participant_id'] = participant_id
    cluster_coef_dict['task'] = task

    return cluster_coef_dict


def calc_participation_coef(file, participant_id, task, use_louvain=False):
    # read the file
    adj_df = _read_adj_matrix(file)
    
    # rename and sort the columns
    adj_df = _sort_columns(adj_df)
    
    adj_r_df = _fishers_z_to_r(adj_df)

    orig_ci = adj_r_df.columns.str.split('-', n=1, expand=True).get_level_values(0)
    if use_louvain:
        np.fill_diagonal(adj_r_df.values, 0)
        ci, _ = community_louvain(adj_r_df.values, B='negative_asym', gamma=1.0)
    else:
        ci = orig_ci

    nan_rois, num_nan_rois, (pos_p_coef, neg_p_coef) = _run_graph_theory_measure(
        adj_r_df, participation_coef_sign, ci=ci,
    )
    # combine participation coefs with their respective rois
    track_idx = 0
    participation_coef_dict = {}
    for roi in adj_r_df.columns:
        roi_pos = roi + '_pos'
        roi_neg = roi + '_neg'
        if roi in list(nan_rois):
            participation_coef_dict[roi_pos] = np.nan
            participation_coef_dict[roi_neg] = np.nan
        else:
            participation_coef_dict[roi_pos] = pos_p_coef[track_idx]
            participation_coef_dict[roi_neg] = neg_p_coef[track_idx]
            track_idx += 1
    
    participation_coef_dict['nan_rois'] = nan_rois
    participation_coef_dict['num_nan_rois'] = num_nan_rois
    participation_coef_dict['participant_id'] = participant_id
    participation_coef_dict['task'] = task

    return participation_coef_dict