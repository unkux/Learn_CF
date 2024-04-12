#!/usr/bin/env python3
# env -S python3 -u

from utils import progress, flatten, set_verbose, vprint
from sklearn.preprocessing import (
    PolynomialFeatures, MaxAbsScaler, PowerTransformer,
    RobustScaler, QuantileTransformer, StandardScaler
)
from sklearn.base import TransformerMixin
from bounded_pool import BoundedPool
from collections import defaultdict
from functools import partial
from copy import deepcopy
import scipy.io as sio
import networkx as nx
import settings as st
import pandas as pd
import numpy as np
import joblib
import math
import fire
import os

# register features
FEATURES = ['R', 'X', 'B', 'flow', 'load', 'max_pred_load', 'capacity',
            'redundancy_ratio', 'redundant_cap', 'betweenness', 'connectivity', 'comb_ratio', 'deg', 'gen',
            'Va0', 'Pd0', 'Qd0', 'Pg0', 'Va1', 'Pd1', 'Qd1', 'Pg1']

# NOTE: transform MultiDiGraph to DiGraph by inserting dummy nodes to split the extra multi-edges
def trans_multi_graph(G):
    Gc = deepcopy(G)
    for e in G.edges:
        if e[2] > 0:
            Gc.remove_edge(*e)
            # NOTE: if transformation is on bi-directional graph, two dummy nodes are created
            n = 'n%d_%d_%d' % e
            d = dict(flow=G.edges[e]['flow'],
                     capacity=G.edges[e]['capacity'])
            Gc.add_edges_from([(e[0], n, d), (n, e[1], d)])
    return nx.DiGraph(Gc)

# NOTE: maximum_flow_value doesn't support MultiDiGraph
def redundant_cap(Gd, e):
    """The maximum flow that can be transmitted from u to v over the residual network excluding link (u, v)
    """
    Gc = deepcopy(Gd)
    et, et_r = e[:2], (e[1], e[0])
    if e[2] > 0:
        # NOTE: two dummy nodes for the bi-directional edges
        et = (e[0], 'n%d_%d_%d' % e)
        et_r = ('n%d_%d_%d' % (e[1], e[0], e[2]), e[0])
    for e1 in [et, et_r]:
        Gc.edges[e1]['capacity'] = 0

    # compute the redundant capacity
    rcap = nx.maximum_flow_value(Gc, e[0], e[1])
    return rcap if rcap > 0 else 1e-6

def find_multi_edges(G):
    multi_edges = defaultdict(lambda: 1)
    for e in G.edges:
        if len(e) == 3 and e[2] > 0:
            multi_edges[e[:2]] += 1
    return dict(multi_edges)

def build_edge_idx(brches, multi_edges):
    edge_idx = {}
    eid = defaultdict(int)
    for i, br in enumerate(brches):
        if br in multi_edges:
            e = (*br, eid[br])
            eid[br] += 1
        else:
            e = (*br, 0)
        edge_idx[e] = i
    return edge_idx

def max_pred_load(e_out, G, lodf, edge_idx):
    """The predicted maximum load in system after the failure of a single link
    """
    load = np.zeros(len(G.edges))
    i_eo = edge_idx[e_out]
    flow_out = G.edges[e_out]['flow']
    for e in G.edges:
        i = edge_idx[e]
        # NOTE: LODF contains invalid values, and diag(lodf) = -1
        v = lodf[i, i_eo]
        load[i] = (G.edges[e]['flow'] + v * flow_out) / G.edges[e]['capacity']
    return load.max()

def make_full_bd_graph(G):
    Gc = deepcopy(G)
    for e in G.edges:
        e_r = (e[1], e[0], e[2])  # e[::-1]
        if e_r not in G.edges:
            Gc.add_edge(
                *e_r,
                # NOTE: set the flow to 0
                flow=0,  # -G.edges[e]['flow'],
                capacity=G.edges[e]['capacity'],)
    return Gc

# first col as key, the rest as values
def mat2dict(m):
    return defaultdict(lambda: [0]*(m.shape[1]-1), zip(m[:, 0], m[:, 1:].tolist()))

def load_data(mat_file, edge_attr):
    # NOTE: take care of the index of data
    mat = sio.loadmat(mat_file)
    lodf = mat['LODF']
    vprint('Number of branches:', len(lodf))

    flow = mat['ps']['branch'][0, 0][:, [0, 1, *edge_attr.values()]]

    # 0: bus id, 1: bus angle (Va)
    va = mat['ps']['bus'][0, 0][:, [0, 8]]
    # NOTE: angle has negative value
    #va[va[:, 1]<0, 1] += 360
    bus_va = mat2dict(va)
    # 0: bus id, 1: real power (Pd), 2: reactive power (Qd)
    shunt = mat2dict(mat['ps']['shunt'][0, 0][:, [0, 1, 2]])
    # 0: bus id, 1: real power (Pg)
    gen = mat2dict(mat['ps']['gen'][0, 0][:, [0, 1]])

    bus = {}
    for e in flow[:, [0, 1]].astype(int):
        # NOTE: ['Va0', 'Pd0', 'Qd0', 'Pg0', 'Va1', 'Pd1', 'Qd1', 'Pg1']
        bus[tuple(e)] = list(flatten([d[n] for n in e for d in [bus_va, shunt, gen]]))

    data = pd.DataFrame(flow, columns=['f', 't', *edge_attr.keys()]).astype({'f': int, 't': int})
    return data, lodf, gen, bus

def find_outliers(df, method='iqr', col='redundancy_ratio'):
    percentile = {'iqr': (0.25, 0.75), 'winsorization': (0.01, 0.99)}
    p1, p2 = percentile[method]
    df_c = df[col]
    lo, hi = df_c.quantile(p1), df_c.quantile(p2)
    if method == 'irq':
        delta = (hi - lo) * 1.5
        lo, hi = lo - delta, hi + delta
    return df.index[(df_c < lo) | (df_c > hi)]

# NOTE: line_dist based on line_graph (dual)
def compute_line_dist(Gl, ed):
    N = len(Gl.nodes)
    ld = np.zeros((N, N))
    lp = nx.shortest_path_length(Gl)
    for f, d in lp:
        fi = ed[f]
        for t, v in d.items():
            ld[fi, ed[t]] = v
    return ld

def build_features(instance, max_workers, insert=False):
    infile = instance + '/ig_data.mat'
    if not os.path.exists(infile):
        raise SystemExit('Error: %s not existed!' % infile)

    # NOTE: MATLAB params index
    edge_attr = dict(R=2, X=3, B=4, capacity=6, flow=11)
    data, lodf, gen, bus = load_data(infile, edge_attr)
    lodf[~np.isfinite(lodf)] = 0

    # NOTE:
    #   - Multiple edges exist between two endpoints, MultiDiGraph is used here
    #   - Edge is stored as a triple
    #   - Negative / bi-directional flow exists in the power grid
    G = nx.from_pandas_edgelist(data, 'f', 't', edge_attr=list(edge_attr.keys()), create_using=nx.MultiDiGraph)

    # NOTE: bi-direction flow for computing redundant_cap
    Gc = make_full_bd_graph(G)

    # Create DiGraph of MultiDiGraph Gc, and compute residual graph
    Gd = trans_multi_graph(Gc)
    for e in Gd.edges:
        Gd.edges[e]['capacity'] -= abs(Gd.edges[e]['flow'])
        # avoid noisy information
        Gd.edges[e]['flow'] = 0

    # keep the order of branches in MATLAB
    brches = data[['f', 't']].to_records(index=False).tolist()
    multi_edges = find_multi_edges(G)
    if multi_edges:
        vprint('Multi-Edges in MultiDiGraph:\n', multi_edges)
    edge_idx = build_edge_idx(brches, multi_edges)

    # dual graph
    Gl = nx.line_graph(G.to_undirected(), create_using=nx.MultiGraph)
    # NOTE: add nodes corresponding to the direct edges (e.g., case2383wp)
    for e in G.edges:
        e0 = (e[1], e[0], 0)
        if e0 in G.edges and e not in Gl.nodes:
            Gl.add_node(e)
            Gl.add_edges_from([(e, el[1]) for el in Gl.edges(e0)])

    # G.to_undirected change the original order of an edge
    ed = edge_idx.copy()
    degree = np.zeros(len(ed))
    for e in Gl.nodes:
        if e not in ed:
            ed[e] = ed[(e[1], e[0], e[2])]
        degree[ed[e]] = Gl.degree(e)

    # whether the line connected to generator
    gen_line = np.zeros(len(edge_idx))
    bus_data = np.zeros((len(edge_idx), 8))
    for e, i in edge_idx.items():
        gen_line[i] = sum(n in gen for n in e[:2])
        bus_data[i] = bus[e[:2]]

    bus_fts = list(zip(['Va0', 'Pd0', 'Qd0', 'Pg0', 'Va1', 'Pd1', 'Qd1', 'Pg1'], bus_data.T))
    fts = [('deg', degree), ('gen', gen_line)] + bus_fts
    if insert:
        insert_feature(instance, fts)
        return None, None, None

    # line distance
    ld = compute_line_dist(Gl, ed)

    vprint('Compute max_pred_load...')
    pg = progress(len(G.edges))
    for e in G.edges:
        pg.update()
        G.edges[e]['max_pred_load'] = max_pred_load(e, G, lodf, edge_idx)

    vprint('Compute redundant_cap, connectivity...')
    with BoundedPool(max_workers) as pool:
        def _update_res(fut, G, e, k):
            pg.update()
            G.edges[e][k] = fut.result()

        for e in G.edges:
            #fp = partial(_update_res, G=G, e=e, k='max_pred_load')
            #pool.submit(max_pred_load, e, G, lodf, edge_idx).add_done_callback(fp)

            fp = partial(_update_res, G=G, e=e, k='redundant_cap')
            pool.submit(redundant_cap, Gd, e).add_done_callback(fp)
            fp = partial(_update_res, G=G, e=e, k='connectivity')
            pool.submit(nx.edge_connectivity, Gc, *e[:2]).add_done_callback(fp)

    vprint('Compute betweenness...')
    betweenness = nx.edge_betweenness_centrality(Gc)

    vprint('Compute other physical/topological features...')
    for e in G.edges:
        pg.update()
        # topological
        G.edges[e]['betweenness'] = betweenness[e]
        # physical
        G.edges[e]['flow'] = abs(G.edges[e]['flow'])
        G.edges[e]['load'] = G.edges[e]['flow'] / G.edges[e]['capacity']
        G.edges[e]['redundancy_ratio'] = G.edges[e]['flow'] / G.edges[e]['redundant_cap']
        G.edges[e]['comb_ratio'] = math.sqrt(G.edges[e]['redundancy_ratio']**2 + G.edges[e]['max_pred_load']**2)

    # NOTE: For MultiDiGraph, add edge_key to distinguish multiple-edges
    df = nx.to_pandas_edgelist(G, edge_key='ekey').set_index(['source', 'target', 'ekey'])
    brches_n = sorted(edge_idx, key=edge_idx.get)
    df = df.loc[brches_n].droplevel('ekey')

    for n, f in fts:
        df[n] = f

    return df, ld, lodf

# NOTE: pre-register the new features in FEATURES
def insert_feature(instance, fts):
    ftfile = instance + '/features.npy'
    ftkeys = ['source', 'target'] + FEATURES
    if os.path.exists(ftfile):
        data = np.load(ftfile)
        if data.shape[1] < len(ftkeys):
            df = pd.DataFrame(data, columns=ftkeys[:data.shape[1]])
            new_cols = []
            for n, f in fts:
                if n not in df.columns and n in ftkeys:
                    df[n] = f
                    new_cols.append(n)
            if new_cols:
                np.save(ftfile, df.to_numpy())
                vprint(f'Insert new data {new_cols}')

def heatmap_comp(*dfs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axes = plt.subplots(1, len(dfs), figsize=(20, 8))
    for i, df in enumerate(dfs):
        sns.heatmap(df, ax=axes[i])  # , annot=True)
    plt.tight_layout()
    plt.show()

def high_corr_filter(df, size_f_org=0, threshold=0.99, plot=False):
    # sort features by overall correlation
    def index_sort(corr):
        return corr.sum().sort_values().index
    df_corr = df.corr().abs()
    index = index_sort(df_corr.iloc[:, :size_f_org]).append(index_sort(df_corr.iloc[:, size_f_org:]))
    df_corr = df_corr.loc[index, index]

    # remove highly-correlated features
    df_triu = df_corr.where(np.triu(np.ones_like(df_corr, dtype=bool), k=1))
    hc_fts = [c for c in df_triu.columns if any(df_triu[c] >= threshold)]
    vprint(f'Highly correlated features:\n{sorted(hc_fts)}')
    df_new = df.drop(hc_fts, axis=1)
    vprint(f'Feature filtered from {df.shape} to {df_new.shape}')

    if plot:
        heatmap_comp(df_corr, df_new[index.drop(hc_fts)].corr().abs())
    return df_new, hc_fts

def boxplot_features(df, keys=None, keys_org=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    keys = list(df.columns) if keys is None else keys
    keys_org = keys if keys_org is None else keys_org
    vsplit = 2
    plt.rc('font', size=16)
    fig, axes = plt.subplots(round(len(keys_org)/vsplit+0.001), vsplit, figsize=(32, 18))
    for i, k in enumerate(keys_org):
        ax = axes.flat[i]
        ax.set_xlabel(k, fontdict={'weight': 'bold'})
        if k in keys:
            sns.boxplot(
                df.iloc[:, keys.index(k)], ax=ax, orient='h', width=0.5, linewidth=4,
                flierprops=dict(marker='o', markerfacecolor='r', markersize=10, linestyle='none'))
        else:
            plt.setp(ax, xticks=[], yticks=[])
    plt.subplots_adjust(hspace=2.5, wspace=0.1, left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

# take care outliers for the features
def build(instance, select=None, poly_fts=False, corr=0.9, positive=False, scaler='max_abs',
          test=None, max_workers=0, insert=False, output='', rerun=False, plot=False, verbose=0):
    set_verbose(verbose)
    ftfile = instance + '/features.npy'
    ldfile = instance + '/line_dist.npz'
    INDEX = ['source', 'target']
    # NOTE: x b can be negative
    if select is None:
        select = FEATURES

    features = list(select)
    if not insert and not rerun and os.path.exists(ftfile) and os.path.exists(ldfile):
        vprint(f'Load existing features {ftfile}', level=1)
        data = np.load(ftfile)
        df = pd.DataFrame(data, columns=INDEX+FEATURES).set_index(INDEX)
        vprint(f'Load existing line_dist {ldfile}', level=1)
        with np.load(ldfile) as ldata:
            ld = ldata['ld']
        mat_file = instance + '/ig_data.mat'
        mat = sio.loadmat(mat_file)
        lodf = mat['LODF']
        lodf[~np.isfinite(lodf)] = 0
        np.fill_diagonal(lodf, 0)

    else:
        df, ld, lodf = build_features(instance, max_workers, insert)
        if insert:
            return None, None
        df = df[FEATURES]
        np.save(ftfile, df.reset_index().to_numpy())
        np.savez_compressed(ldfile, ld=ld)

    df = df[features]

    if verbose > 2:
        print(df.reset_index())
        print(df.describe().drop('count'))

    # rethink max_pred_load > 1, which predicates the line outage
    for f in ['redundancy_ratio', 'comb_ratio']:
        if f in features:
            df.loc[df[f] > 1, f] = np.log(df.loc[df[f] > 1, f]) + 1

    df[df.abs() <= 1e-6] = 0

    if verbose > 10:
        # engineered features
        for f in ['max_pred_load', 'redundancy_ratio', 'redundant_cap', 'comb_ratio']:
            idx = find_outliers(df, method='winsorization', col=f)
            if not idx.empty:
                print(f'\nOutliers w.r.t. {f}')
                print(df.loc[idx])

    if positive:
        vprint('Apply positive feature!')
        df = df.abs()
        lodf = np.abs(lodf)

    # Dataframe indexing is quite slow, return directly ndarray
    # NOTE: np array returned from pandas is not directly supported by numba-dpex for intel GPU,
    # due to the order of memory layout, ndarray.copy has a default C-order.
    data = df.values.copy()
    size_f_org = data.shape[1]
    keys_f_org = df.columns.tolist()

    if plot:
        boxplot_features(df)

    # polynomial expansion
    if poly_fts:
        poly = PolynomialFeatures(2, include_bias=False)
        data = poly.fit_transform(data)

    # NOTE: consider the filter and scale when testing other models
    message = 'for generalization test!'
    if test:
        if os.path.exists(str(test)):
            if scaler and isinstance(scaler, dict):
                mscaler = scaler
            else:
                vprint(f'Load feature filter & scale {test} {message}', level=1)
                mscaler = joblib.load(test)
            if mscaler:
                ld_max, lodf_maxa, bad_fts_id, scaler = (mscaler[k] for k in ['ld', 'lo', 'bf', 'scaler'])
                data = np.delete(data, bad_fts_id, axis=1)
        else:
            raise SystemExit(f'Error: {test} not exists!')
    else:
        df = pd.DataFrame(data, index=df.index)
        # filter low variance features
        std = df.std()
        lv_fts = std[std <= 1e-6].index
        vprint(f'Low variance features: {lv_fts.tolist()}')
        df.drop(lv_fts, axis=1, inplace=True)
        size_f_org -= sum(lv_fts < size_f_org)

        # filter highly correlated features
        df, hc_fts = high_corr_filter(df, size_f_org, threshold=corr, plot=plot)
        bad_fts_id = lv_fts.tolist() + hc_fts
        data = df.values.copy()

        ld_max, lodf_maxa = ld.max(), np.abs(lodf).max()

        if isinstance(scaler, str):
            if scaler == 'max_abs':
                scaler = MaxAbsScaler().fit(data)
            elif scaler == 'power':
                scaler = PowerTransformer(method='yeo-johnson').fit(data)
            elif scaler == 'robust':
                scaler = RobustScaler(quantile_range=(1, 99)).fit(data)
            elif scaler == 'quantile':
                scaler = QuantileTransformer(output_distribution="normal", random_state=42).fit(data)
            elif scaler == 'standard':
                scaler = StandardScaler().fit(data)
            else:
                scaler = None

        mscaler = dict(ld=ld_max, lo=lodf_maxa, bf=bad_fts_id, scaler=scaler)

        if output:
            if isinstance(output, bool):
                output = f"{instance}/{st.D_OUTPUT}"
            os.makedirs(output, exist_ok=True)
            fname = f"{output}/{st.F_FT_SC}" % os.path.basename(instance)
            vprint(f'Save feature scale {fname} {message}', level=1)
            joblib.dump(mscaler, fname)

    # max abs scaler for distance features
    ld, lodf = ld/ld_max, lodf/lodf_maxa
    if isinstance(scaler, TransformerMixin):
        data = scaler.transform(data)
    else:
        scaler = None
        print('No valid scaler, use orginal data!')

    if plot:
        keys = [f for i, f in enumerate(keys_f_org) if i not in bad_fts_id]
        df = pd.DataFrame(data[:, :len(keys)], index=df.index)
        boxplot_features(df, keys=keys, keys_org=keys_f_org)

    ft_l1 = np.abs(data).sum(axis=1)
    vprint(f'||feature||_1: [{ft_l1.min()}, {ft_l1.max()}]')
    vprint(f'No. of base features: {data.shape[1]}')

    return data, np.stack([ld, lodf]), mscaler


if __name__ == '__main__':
    # Not use pager for printing help text
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(build, serialize=lambda results: None)
