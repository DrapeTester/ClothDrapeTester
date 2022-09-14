import sys
import numpy as np
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter


def load_data(idx_file, data_dir):
    data_file = osp.join(data_dir, 'dataset.csv')
    with open(idx_file, 'r') as f:
        indices = [int(i) for i in f.readlines()]
    data_lst = []
    with open(data_file, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            while line.find(" ") != -1:
                line = line.replace(" ", "")
            data_line = [float(i) for i in line.split(",")]
            data_lst.append(data_line)
    selected_data_lst = itemgetter(*indices)(data_lst)
    density_lst = []
    prop_lst = []

    for i in selected_data_lst:
        density_lst.append(i[0])
        prop_lst.append(i[1:])
    return density_lst, prop_lst


def apply_log(scalar):
    return (1 if scalar > 0 else -1) * np.log(np.abs(scalar))


def apply_log_on_data(data):
    '''
    warp, weft, bias, warp, weft, bias (nonlinear entries)
    '''
    assert len(data) == 6
    for i in range(len(data)):
        data[i] = apply_log(data[i])
    return data


class FabricDataset(Dataset):
    def __init__(self, data_dir, phase_train, data_aug=10) -> None:
        train_file = "train.txt"
        test_file = "test.txt"
        self.feature_dim = 6
        # load data
        if phase_train == True:
            self.density_lst, self.prop_lst = load_data(train_file, data_dir)
        else:
            self.density_lst, self.prop_lst = load_data(test_file, data_dir)

        # data preprocess: apply log
        for idx in range(len(self.prop_lst)):
            # print(f"raw {self.prop_lst[idx]}")
            self.prop_lst[idx] = apply_log_on_data(self.prop_lst[idx])
            # print(self.prop_lst[idx])
            # exit()

        # data_augmentation by apply gaussian noise
        sigma = 0.05
        mu = 0
        num_of_old_data = len(self.prop_lst)
        for _ in range(data_aug):
            for idx in range(num_of_old_data):
                new_data = self.prop_lst[idx] + np.random.randn(
                    len(self.prop_lst[idx])) * sigma + mu
                self.prop_lst.append(new_data)
                self.density_lst.append(self.density_lst[idx])


        # convert to float
        for i in range(len(self.prop_lst)):
            self.prop_lst[i] = np.array(self.prop_lst[i], dtype=np.float32)
            self.density_lst[i] = float(self.density_lst[i])

    def __len__(self):
        return len(self.prop_lst)

    def __getitem__(self, index):
        return self.prop_lst[index], 1

    def get_data(self):
        return self.prop_lst


def get_dataloader(data_dir, batch_size=32, data_scale=10):
    train_set = FabricDataset(data_dir=data_dir,
                              phase_train=True,
                              data_aug=data_scale)
    test_set = FabricDataset(data_dir=data_dir,
                             phase_train=False,
                             data_aug=data_scale)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    return test_loader, train_loader


if __name__ == "__main__":
    # 1. load dataset
    root_datadir = "..\subspace"
    train_loader, test_loader = get_dataloader(root_datadir)

    # 2. VAE structure

    # 3. VAE training

    # 4. save weight
