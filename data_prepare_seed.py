import os
import scipy.io as sio
import numpy as np


def feature_wrap(feature_3d):
    feature_2d = np.zeros((feature_3d.shape[1], 310))
    for i in range(feature_3d.shape[1]):
        feature_2d[i, :] = feature_3d[:, i, :].reshape(-1)
    return feature_2d


os.chdir('your_path')
label_mat = sio.loadmat('label.mat')
label = label_mat['label'].flatten()

for i in range(1, 16):
    os.chdir('your_path')
    # os.chdir('')
    folder_list = sorted([f for f in os.listdir() if f.endswith('.mat')])

    file_name_session_1 = folder_list[i * 3 - 3]
    file_name_session_2 = folder_list[i * 3 - 2]
    file_name_session_3 = folder_list[i * 3 - 1]

    data_session_1 = sio.loadmat(file_name_session_1)
    data_session_2 = sio.loadmat(file_name_session_2)
    data_session_3 = sio.loadmat(file_name_session_3)

    feature_session_1, label_session_1 = [], []
    feature_session_2, label_session_2 = [], []
    feature_session_3, label_session_3 = [], []

    for j in range(1, 16):
        feature_session_1.append(feature_wrap(data_session_1[f'de_LDS{j}']))
        label_session_1.append(np.zeros(data_session_1[f'de_LDS{j}'].shape[1]) + label[j - 1])

        feature_session_2.append(feature_wrap(data_session_2[f'de_LDS{j}']))
        label_session_2.append(np.zeros(data_session_2[f'de_LDS{j}'].shape[1]) + label[j - 1])

        feature_session_3.append(feature_wrap(data_session_3[f'de_LDS{j}']))
        label_session_3.append(np.zeros(data_session_3[f'de_LDS{j}'].shape[1]) + label[j - 1])

    feature_session_1 = np.vstack(feature_session_1)
    label_session_1 = np.hstack(label_session_1) + 1

    feature_session_2 = np.vstack(feature_session_2)
    label_session_2 = np.hstack(label_session_2) + 1

    feature_session_3 = np.vstack(feature_session_3)
    label_session_3 = np.hstack(label_session_3) + 1

    dataset_session1 = {'feature': feature_session_1, 'label': label_session_1}
    dataset_session2 = {'feature': feature_session_2, 'label': label_session_2}
    dataset_session3 = {'feature': feature_session_3, 'label': label_session_3}

    output_dir = '/home/user/disk2/eeg/SEED/feature'
    os.makedirs(output_dir, exist_ok=True)

    sub = f'sub_{i}_'
    sio.savemat(os.path.join(output_dir, f'{sub}session_1.mat'), {'dataset_session1': dataset_session1})
    sio.savemat(os.path.join(output_dir, f'{sub}session_2.mat'), {'dataset_session2': dataset_session2})
    sio.savemat(os.path.join(output_dir, f'{sub}session_3.mat'), {'dataset_session3': dataset_session3})
