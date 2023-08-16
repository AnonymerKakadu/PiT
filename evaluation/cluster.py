import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import patches, pyplot as plt
from scipy.optimize import linear_sum_assignment
from fau_colors import cmaps

colors = cmaps.faculties_all


def k_means(fe, cl):
    kmeans = KMeans(n_clusters=cl, random_state=0).fit(fe)
    return kmeans.labels_


def best_matching_accuracy(predicted_clusters, gt_clusters):
    num_samples = len(predicted_clusters)
    mask = predicted_clusters != -1
    predicted_clusters = predicted_clusters[mask]
    unique_pred_clusters = np.unique(predicted_clusters)
    unique_gt_clusters = np.unique(gt_clusters)
    num_predicted_clusters = np.size(unique_pred_clusters)
    num_gt_clusters = np.size(unique_gt_clusters)
    # print(unique_gt_clusters, np.arange(num_gt_clusters))
    if not np.all(np.arange(num_predicted_clusters) == unique_pred_clusters):
        for i, cluster in enumerate(unique_pred_clusters):
            predicted_clusters[predicted_clusters == cluster] = i
    unique_pred_clusters = np.unique(predicted_clusters)
    assert np.all(np.arange(num_predicted_clusters) == unique_pred_clusters)
    if not np.all(np.arange(num_gt_clusters) == unique_gt_clusters):
        for i, cluster in enumerate(unique_gt_clusters):
            gt_clusters[gt_clusters == cluster] = i
    unique_gt_clusters = np.unique(gt_clusters)
    assert np.all(np.arange(num_gt_clusters) == unique_gt_clusters)
    gt_clusters = gt_clusters[mask]
    weight_matrix = np.empty((num_predicted_clusters, num_gt_clusters))
    for cluster_id in unique_pred_clusters:
        for gt_cluster_id in unique_gt_clusters:
            weight_matrix[cluster_id, gt_cluster_id] = np.sum((predicted_clusters == cluster_id) & (gt_clusters == gt_cluster_id))
            pred_indices, gt_indices = linear_sum_assignment(weight_matrix, maximize=True)
    return pred_indices, gt_indices, weight_matrix[pred_indices, gt_indices].sum() / num_samples


base = []
for ver in ['color_back', 'color_pad', 'bw_back', 'bw_pad']:
    for z in [['Nürnberg', 0, 1], ['Berlin', 2, 3], ['Schönbrunn', 4, 5], ['Mulhouse', 6, 7]]:
        features = np.loadtxt(f'/media/richard/Richard/ReID_Result/domain/{ver}/{z[0]}/features_pb_{z[0]}')
        if ver[-3:] == 'pad':
            label_file = pd.read_csv('/media/richard/Richard/ReID_Datasets/10_all_pad/track_info.csv')
        else:
            label_file = pd.read_csv('/media/richard/Richard/ReID_Datasets/9_all_back/track_info.csv')
        labels = label_file[label_file['id'].isin(z[1:])]['id'].values
        predictions = k_means(features, 2)
        _, _, acc = best_matching_accuracy(predictions, labels)
        print(acc)
        base.append([ver, z[0], acc])

# load data
col = ['version', 'domain', 'acc']
base = pd.DataFrame(base, columns=col)
zoos = ['Nürnberg', 'Berlin', 'Schönbrunn', 'Mulhouse']
legend_patches = []
c_counter = [4, 5, 6, 7]

# def graph
_, ax = plt.subplots()
plt.ylabel('Accuracy (%)')
plt.xlabel("Datasets")
ax.set_ylim([0.0, 1.0])
ax.set_yticks([i / 100 for i in range(0, 101, 10)])
ax.set_yticklabels(range(0, 101, 10))
ax.set_axisbelow(True)
plt.grid(axis='y')
ax.set_xticks(range(4))
ax.set_xticklabels(zoos)

# set data
off = np.array([-0.3, -0.1, 0.1, 0.3])
i = np.array(range(4))

for idx, v in enumerate(['color_back', 'color_pad', 'bw_back', 'bw_pad']):
    data = base[base['version'] == v]
    assert list(data['domain'].values) == zoos
    plt.bar(i + off[idx], data['acc'], align='center', width=0.2, color=colors[c_counter[idx]])
    legend_patches.append(patches.Patch(color=colors[c_counter[idx]], label=f'{v}'))

plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(right=0.78)

plt.show()
