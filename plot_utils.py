import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import scipy as sp
import itertools
import numpy as np
import pandas as pd


def plot_dataset_correlation(dataset, map_df, metric_df, bbox_metric_df, x_label = 'Performance', y_label = 'Transferability Metric'):
    fig, axes = plt.subplots(1, 2, figsize = (18,7))
    layer_color = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:orange']

    map_list = list(map_df.loc[dataset][0:5])[::-1] # Reverse because fine tuning 0 layer should correlate with features of layer 5.
    # map_df column represent the number of layers being fine tuning (0 meaning head only) as metric_df columns represent the metric with features extracted from layer i
    feats_metric = metric_df.loc[dataset]
    bbox_feats_metric = bbox_metric_df.loc[dataset]

    custom_lines = [Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:blue', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:green', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:purple', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:red', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:orange', markersize=8)]

    r, p = sp.stats.pearsonr(map_list, feats_metric)
    r_spearman, p_spearman = sp.stats.spearmanr(map_list, feats_metric)
    r_kendall, p_kendall = sp.stats.kendalltau(map_list, feats_metric)

    # Create legend for correlation
    line_pearson, = axes[0].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
    line_spearman,=axes[0].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
    line_kendall, =axes[0].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))
    correlation_legend = axes[0].legend(handles=[line_pearson, line_spearman, line_kendall], loc='upper right')
    sns.regplot(x = map_list,  y = feats_metric, fit_reg= False, ax = axes[0], scatter_kws={'color':list(layer_color)})

    #Add legends
    axes[0].add_artist(correlation_legend)
    axes[0].legend(custom_lines, ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'], loc = 'lower left')

    axes[0].set_title(f"{y_label} VS {x_label} with duplicated features")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)

    r, p = sp.stats.pearsonr(map_list, bbox_feats_metric)
    r_spearman, p_spearman = sp.stats.spearmanr(map_list, bbox_feats_metric)
    r_kendall, p_kendall = sp.stats.kendalltau(map_list, bbox_feats_metric)

    # Create legend for correlation
    line_pearson, = axes[1].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
    line_spearman,=axes[1].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
    line_kendall, =axes[1].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))
    correlation_legend = axes[1].legend(handles=[line_pearson, line_spearman, line_kendall], loc='upper right')
    sns.regplot(x = map_list,  y = bbox_feats_metric, fit_reg= False, ax = axes[1], scatter_kws={'color':list(layer_color)})

    #Add legends
    axes[1].add_artist(correlation_legend)
    axes[1].legend(custom_lines, ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'], loc = 'lower left')

    axes[1].set_title(f"{y_label} VS {x_label} with bbox features")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    fig.suptitle(f"{dataset} DATASET");


def plot_all_datasets_correlation( map_df, metric_df, bbox_metric_df, x_label = 'Performance', y_label = 'Transferability Metric'):
    fig, axes = plt.subplots(1, 2, figsize = (18,7))
    layer_color = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:orange']

    map_list = map_df.iloc[:,0] # Performance of retraining head only

    custom_lines = [Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:blue', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:green', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:purple', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:red', markersize=8),
                            Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='tab:orange', markersize=8)]

    metric_list = metric_df.iloc[:,4]
    bbox_metric_list = bbox_metric_df.iloc[:,4]

    r, p = sp.stats.pearsonr(map_list, metric_list)
    r_spearman, p_spearman = sp.stats.spearmanr(map_list, metric_list)
    r_kendall, p_kendall = sp.stats.kendalltau(map_list, metric_list)
    line_pearson, = axes[0].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
    line_spearman,=axes[0].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
    line_kendall, =axes[0].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))
    correlation_legend = axes[0].legend(handles=[line_pearson, line_spearman, line_kendall], loc='upper right')
    axes[0].add_artist(correlation_legend)
    sns.regplot(x = map_list,  y = metric_list, fit_reg= False, ax = axes[0], scatter_kws={'color':list(layer_color)})
    axes[0].legend(custom_lines, ['BCCD', 'CHESS', 'Global_Wheat', 'VOC', 'Open_Images'], loc = 'lower left')
    axes[0].set_title(f"{y_label} VS {x_label} with duplicated features")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)

    r, p = sp.stats.pearsonr(map_list, bbox_metric_list)
    r_spearman, p_spearman = sp.stats.spearmanr(map_list, bbox_metric_list)
    r_kendall, p_kendall = sp.stats.kendalltau(map_list, bbox_metric_list)
    line_pearson, = axes[1].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
    line_spearman,=axes[1].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
    line_kendall, =axes[1].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))
    correlation_legend = axes[1].legend(handles=[line_pearson, line_spearman, line_kendall], loc='upper right')
    axes[1].add_artist(correlation_legend)
    sns.regplot(x = map_list,  y = bbox_metric_list, fit_reg= False, ax = axes[1], scatter_kws={'color':list(layer_color)})
    axes[1].set_title(f"{y_label} VS {x_label} with bbox features")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    fig.suptitle('Correlation for different datasets')


def plot_all_synth( map_df, metric_df, bbox_metric_df, selection = 'source', x_label = 'Performance', y_label = 'Transferability Metric'):
    datasets = ['MNIST', 'KMNIST', 'EMNIST', 'FASHION_MNIST', 'USPS']
    for dataset in datasets : 

        if selection == 'target':
            map_list = map_df.loc[dataset, map_df.columns != dataset]
            metric_list = metric_df.loc[dataset, map_df.columns != dataset]
            bbox_metric_list = bbox_metric_df.loc[dataset, map_df.columns != dataset]
            dataset_names = map_df.loc[dataset, map_df.columns != dataset].index.values
        elif selection == 'source': 
            map_list = map_df.loc[map_df.columns != dataset, dataset]
            metric_list = metric_df.loc[map_df.columns != dataset, dataset]
            bbox_metric_list = bbox_metric_df.loc[map_df.columns != dataset, dataset]
            dataset_names = map_df.loc[map_df.columns != dataset, dataset].index.values

        layer_color = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red']

        fig, axes = plt.subplots(1, 2, figsize = (18,7))

        custom_lines = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='tab:blue', markersize=8),
                                Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='tab:green', markersize=8),
                                Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='tab:purple', markersize=8),
                                Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='tab:red', markersize=8),
                                Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='tab:orange', markersize=8)]

        # Compute correlation metrics
        r, p = sp.stats.pearsonr(map_list, metric_list)
        r_spearman, p_spearman = sp.stats.spearmanr(map_list, metric_list)
        r_kendall, p_kendall = sp.stats.kendalltau(map_list, metric_list)

        # Create legend for correlation
        line_pearson, = axes[0].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
        line_spearman,=axes[0].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
        line_kendall, =axes[0].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))
        correlation_legend = axes[0].legend(handles=[line_pearson, line_spearman, line_kendall], loc='upper right')

        #Scatter plot
        sns.regplot(x = map_list,  y = metric_list, fit_reg= False, ax = axes[0], scatter_kws={'color':list(layer_color)})

        #Add legends
        axes[0].add_artist(correlation_legend)
        axes[0].legend(custom_lines,  dataset_names, loc = 'lower left')

        #Add titles and labels
        axes[0].set_title(f"{y_label} VS {x_label} with duplicated features")
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel(y_label)

        # Compute correlation metrics
        r, p = sp.stats.pearsonr(map_list, bbox_metric_list)
        r_spearman, p_spearman = sp.stats.spearmanr(map_list, bbox_metric_list)
        r_kendall, p_kendall = sp.stats.kendalltau(map_list, bbox_metric_list)

        # Create legend for correlation
        line_pearson, = axes[1].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
        line_spearman,=axes[1].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
        line_kendall, =axes[1].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))
        correlation_legend = axes[1].legend(handles=[line_pearson, line_spearman, line_kendall], loc='upper right')

        #Scatter plot
        sns.regplot(x = map_list,  y = bbox_metric_list, fit_reg= False, ax = axes[1], scatter_kws={'color':list(layer_color)})

        #Add legends
        axes[1].add_artist(correlation_legend)
        axes[1].legend(custom_lines,  dataset_names, loc = 'lower left')

        #Add titles and labels
        axes[1].set_title(f"{y_label} VS {x_label} with duplicated features")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel(y_label)
        if selection == 'target' : 
            fig.suptitle(f'Correlation for {selection} selection with a model pretrained on {dataset} ')
        else : 
            fig.suptitle(f'Correlation for {selection} selection with target dataset {dataset} ')


def plot_all_synth_aggregated(map_df, metric_df, bbox_metric_df, x_label = 'Performance', y_label = 'Transferability Metric', coloring = 'target'):
    datasets = ['MNIST', 'KMNIST', 'EMNIST', 'FASHION_MNIST', 'USPS']
    colors = {'MNIST' :'tab:blue', 'KMNIST': 'tab:green', 'EMNIST': 'tab:purple', 'FASHION_MNIST': 'tab:orange', 'USPS':'tab:red'}
    map_list, metric_list, bbox_metric_list = [], [], []
    labels_list, color_list = [], []
    for dataset_source, dataset_target in itertools.permutations(datasets, 2):
        map = map_df.loc[dataset_source, dataset_target]
        map_list.append(map)
        metric = metric_df.loc[dataset_source, dataset_target]
        metric_list.append(metric)
        bbox_metric = bbox_metric_df.loc[dataset_source, dataset_target]
        bbox_metric_list.append(bbox_metric)
        labels_list.append(dataset_source[0]+ ' to ' + dataset_target[0])

        if coloring == 'target':
            color_list.append(colors[dataset_target])
        else :
            color_list.append(colors[dataset_source])

    fig, axes = plt.subplots(1, 2, figsize = (18,7))
    r, p = sp.stats.pearsonr(map_list, metric_list)
    r_spearman, p_spearman = sp.stats.spearmanr(map_list, metric_list)
    r_kendall, p_kendall = sp.stats.kendalltau(map_list, metric_list)

    axes[0].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
    axes[0].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
    axes[0].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))

    axes[0].legend()
    axes[0].scatter(map_list, metric_list, c= color_list)

    axes[0].set_title(f"{y_label} VS {x_label} with duplicated features")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)

    for i in range(len(map_list)):
        axes[0].text(map_list[i]+.005, metric_list[i] + 0.001, labels_list[i])

    r, p = sp.stats.pearsonr(map_list, bbox_metric_list)
    r_spearman, p_spearman = sp.stats.spearmanr(map_list, bbox_metric_list)
    r_kendall, p_kendall = sp.stats.kendalltau(map_list, bbox_metric_list)

    axes[1].plot([], [], ' ', label='r={:.2f}, p={:.2g}'.format(r, p))
    axes[1].plot([], [], ' ', label='rs={:.2f}, p={:.2g}'.format(r_spearman, p_spearman))
    axes[1].plot([], [], ' ', label='rk={:.2f}, p={:.2g}'.format(r_kendall, p_kendall))

    axes[1].legend()
    axes[1].scatter(map_list, bbox_metric_list, c = color_list)

    axes[1].set_title(f"{y_label} VS {x_label} with bbox features")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)

    for i in range(len(map_list)):
        axes[1].text(map_list[i]+.005, bbox_metric_list[i] + 0.001, labels_list[i])


def compute_correlations_synth(map_df, metric_dfs, bbox_metric_dfs, metric_names, selection = 'source', correlation = 'pearson'):
    datasets = map_df.index.values
    correlations = np.zeros((len(datasets), len(metric_names)))
    correlations_bbox = np.zeros((len(datasets), len(metric_names)))
    for column in range(len(metric_names)): 
        metric_df, bbox_metric_df = metric_dfs[column], bbox_metric_dfs[column]
        for row, dataset in enumerate(datasets) : 
            if selection == 'target':
                map_list = map_df.loc[dataset, map_df.columns != dataset]
                metric_list = metric_df.loc[dataset, map_df.columns != dataset]
                bbox_metric_list = bbox_metric_df.loc[dataset, map_df.columns != dataset]
                
            elif selection == 'source': 
                map_list = map_df.loc[map_df.columns != dataset, dataset]
                metric_list = metric_df.loc[map_df.columns != dataset, dataset]
                bbox_metric_list = bbox_metric_df.loc[map_df.columns != dataset, dataset]
            
            if correlation == 'pearson':
                    r, p = sp.stats.pearsonr(map_list, metric_list)
                    r_bbox, p_bbox = sp.stats.pearsonr(map_list, bbox_metric_list)
            elif correlation == 'spearman':
                r ,p = sp.stats.spearmanr(map_list, metric_list)
                r_bbox ,p_bbox = sp.stats.spearmanr(map_list, bbox_metric_list)
            elif correlation =='kendall': 
                r, p = sp.stats.kendalltau(map_list, metric_list)
                r_bbox, p_bbox = sp.stats.kendalltau(map_list, bbox_metric_list)

            correlations[row, column] = r
            correlations_bbox[row, column] = r_bbox
    df_corr = pd.DataFrame(np.round(correlations,2), index = datasets , columns = metric_names)
    df_corr_bbox = pd.DataFrame(np.round(correlations_bbox,2), index = datasets , columns = metric_names)
    df_multi = pd.concat([df_corr, df_corr_bbox], axis=1, keys=['Duplicates', 'Bbox'])

    if selection == 'source': 
        df_multi.index.name= f" Target dataset"
    else: 
        df_multi.index.name= f" Source dataset"
    return df_multi



def compute_correlations_real(map_df, metric_dfs, bbox_metric_dfs, metric_names, correlation = 'pearson'):
    datasets =  map_df.index.values
    correlations = np.zeros((len(datasets), len(metric_names)))
    correlations_bbox = np.zeros((len(datasets), len(metric_names)))
    for column in range(len(metric_names)): 
        metric_df, bbox_metric_df = metric_dfs[column], bbox_metric_dfs[column]

        #Compute correlation for each datasets over layers
        for row, dataset in enumerate(datasets) : 
            map_list = map_df.loc[dataset,:][0:5][::-1]# Reverse because fine tuning 0 layer should correlate with features of layer 5.
            # map_df column represent the number of layers being fine tuning (0 meaning head only) as metric_df columns represent the metric with features extracted from layer i
            metric_list = metric_df.loc[dataset,:]
            bbox_metric_list = bbox_metric_df.loc[dataset,:]
            
            if correlation == 'pearson':
                    r, p = sp.stats.pearsonr(map_list, metric_list)
                    r_bbox, p_bbox = sp.stats.pearsonr(map_list, bbox_metric_list)
            elif correlation == 'spearman':
                r ,p = sp.stats.spearmanr(map_list, metric_list)
                r_bbox ,p_bbox = sp.stats.spearmanr(map_list, bbox_metric_list)
            elif correlation =='kendall': 
                r, p = sp.stats.kendalltau(map_list, metric_list)
                r_bbox, p_bbox = sp.stats.kendalltau(map_list, bbox_metric_list)

            correlations[row, column] = r
            correlations_bbox[row, column] = r_bbox


 
    df_corr = pd.DataFrame(np.round(correlations,2), index = datasets,columns = metric_names)
    df_corr_bbox = pd.DataFrame(np.round(correlations_bbox,2) , index = datasets, columns = metric_names)
    df_multi = pd.concat([df_corr, df_corr_bbox], axis=1, keys=['Duplicates', 'Bbox'])
    df_multi.index.name= f" Target dataset"

    return df_multi

def compute_correlations_from_oi(map_list, metric_lists, bbox_metric_lists, metric_names, correlation = 'pearson'):
    correlations = np.zeros((1, len(metric_names)))
    correlations_bbox = np.zeros((1 , len(metric_names)))
    
    for i in range(len(metric_lists)) : 

        #Compute correlation over datasets
        metric_list = metric_lists[i]
        bbox_metric_list = bbox_metric_lists[i]
        if correlation == 'pearson':
            r, p = sp.stats.pearsonr(map_list, metric_list)
            r_bbox, p_bbox = sp.stats.pearsonr(map_list, bbox_metric_list)
        elif correlation == 'spearman':
            r ,p = sp.stats.spearmanr(map_list, metric_list)
            r_bbox ,p_bbox = sp.stats.spearmanr(map_list, bbox_metric_list)
        elif correlation =='kendall': 
            r, p = sp.stats.kendalltau(map_list, metric_list)
            r_bbox, p_bbox = sp.stats.kendalltau(map_list, bbox_metric_list)

        correlations[-1, i] = r
        correlations_bbox[-1, i] = r_bbox

    index = ['OpenImages BS']
    df_corr = pd.DataFrame(np.round(correlations,2), index = index,columns = metric_names)
    df_corr_bbox = pd.DataFrame(np.round(correlations_bbox,2) , index = index, columns = metric_names)
    df_multi = pd.concat([df_corr, df_corr_bbox], axis=1, keys=['Duplicates', 'Bbox'])
    df_multi.index.name= f" Target dataset"

    return df_multi

def compute_correlations_from_oi_layers(map_df, metric_dfs, bbox_metric_dfs, metric_names, correlation = 'pearson'):

    layers = map_df.index.values
    correlations = np.zeros((len(layers), len(metric_names)))
    correlations_bbox = np.zeros((len(layers) , len(metric_names)))
    
    for column in range(len(metric_names)) : 
        metric_df, bbox_metric_df = metric_dfs[column], bbox_metric_dfs[column]
        for row, layer in enumerate(layers) : 
            map_list = map_df.loc[layer,:]
            metric_list = metric_df.loc[layer,:]
            bbox_metric_list = bbox_metric_df.loc[layer,:]
        #Compute correlation over datasets
            if correlation == 'pearson':
                r, p = sp.stats.pearsonr(map_list, metric_list)
                r_bbox, p_bbox = sp.stats.pearsonr(map_list, bbox_metric_list)
            elif correlation == 'spearman':
                r ,p = sp.stats.spearmanr(map_list, metric_list)
                r_bbox ,p_bbox = sp.stats.spearmanr(map_list, bbox_metric_list)
            elif correlation =='kendall': 
                r, p = sp.stats.kendalltau(map_list, metric_list)
                r_bbox, p_bbox = sp.stats.kendalltau(map_list, bbox_metric_list)

            correlations[row, column] = r
            correlations_bbox[row, column] = r_bbox


    index = layers
    df_corr = pd.DataFrame(np.round(correlations,2), index = index,columns = metric_names)
    df_corr_bbox = pd.DataFrame(np.round(correlations_bbox,2) , index = index, columns = metric_names)
    df_multi = pd.concat([df_corr, df_corr_bbox], axis=1, keys=['Duplicates', 'Bbox'])
    df_multi.index.name= f"Layer"

    return df_multi