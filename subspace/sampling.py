import torch
import pickle as pkl
import numpy as np
from dataloader import get_dataloader
from matplotlib import pyplot as plt
from tqdm import tqdm

subspace_weight = "./vae_subspace.pkl"


def get_data():
    test_loader, train_loader = get_dataloader("../dataset", data_scale=1)
    test_data = test_loader.dataset.get_data()
    train_data = train_loader.dataset.get_data()
    gt_data = np.array(train_data + test_data)
    print(gt_data.shape)
    return gt_data


def get_noise(shape):
    nums = shape[0]
    dims = shape[1]
    noise = []
    for i in range(nums):
        # generate mu and sigma
        mu = np.random.rand() - 0.5
        sigma = np.random.rand() * 0.4 + 0.8

        # create noise
        noise.append(np.random.normal(mu, sigma, size=dims))
    return noise


def calculate_sample_point_from_point_cloud(raw_point_cloud: np.ndarray,
                                            min_dist_thre=2):
    assert type(raw_point_cloud) == np.ndarray, type(raw_point_cloud)
    raw_point_cloud = list(raw_point_cloud)
    sampled_pts = []
    compare_repo = None
    for _idx in tqdm(range(len(raw_point_cloud))):
        raw_pt = raw_point_cloud[_idx]
        if len(sampled_pts) == 0:
            sampled_pts.append(raw_pt)
            compare_repo = np.array([raw_pt])
        else:
            diff = raw_pt - compare_repo

            if diff.shape[1] == 3:
                diff_norm = np.linalg.norm(diff, axis=1)
                min_dist = np.min(diff_norm)
                if min_dist > min_dist_thre:
                    sampled_pts.append(raw_pt)
                    compare_repo = np.concatenate(
                        [compare_repo, np.array([raw_pt])], axis=0)
            elif diff.shape[1] == 6:
                linear_norm = np.linalg.norm(diff[:, :3], axis=1)
                nonlinear_norm = np.linalg.norm(diff[:, 3:], axis=1)
                linear_min_dist = np.min(linear_norm)
                nonlinear_min_dist = np.min(nonlinear_norm)

                if (linear_min_dist > min_dist_thre) or (nonlinear_min_dist >
                                                         min_dist_thre):
                    sampled_pts.append(raw_pt)
                    compare_repo = np.concatenate(
                        [compare_repo, np.array([raw_pt])], axis=0)
                else:
                    pass
            else:
                raise ValueError

    sampled_pts = np.array(sampled_pts)
    return sampled_pts

def get_expanded_range(gt_data):
    gt_min = np.min(gt_data, axis=0)
    gt_max = np.max(gt_data, axis=0)
    gt_range = gt_max - gt_min
    gt_mean = (gt_min + gt_max) / 2
    gt_min = gt_mean - gt_range * 0.55
    gt_max = gt_mean + gt_range * 0.55
    return gt_min, gt_max
    
def remove_out_of_range(gt_data, gt_min, gt_max):
    new_data = []
    for i in range(gt_data.shape[0]):
        if (gt_data[i] < gt_min).any() or (gt_data[i] > gt_max).any() :
            continue
        else :
            new_data.append(gt_data[i])
    return  np.array(new_data)

if __name__ == "__main__":
    sample_scale = 10
    # 1. load VAE weight
    with open(subspace_weight, 'rb') as f:
        model = torch.load(subspace_weight)
    total_param = sum(p.numel() for p in model.parameters())
    print(f"[log] model param {total_param}")

    # drawer = Draw3D()
    cuda_device = torch.device("cuda:0")
    gt_data = get_data()

    # drawer.add_points(list(gt_data[:, 0]),
    #                   list(gt_data[:, 1]),
    #                   list(gt_data[:, 2]),
    #                   label='raw')

    gt_range = get_expanded_range(gt_data)

    gt_data = torch.from_numpy(gt_data).to(cuda_device)
    mu, logvar = model.encode(gt_data)
    bot = model.reparameterize(mu, logvar).detach()

    sampled_data = []
    bot = np.array(bot.cpu())
    for i in range(sample_scale):
        noise = get_noise(np.array(bot.shape))
        bot_noised_comp = torch.from_numpy(
            (bot + noise).astype(np.float32)).to(cuda_device)
        sampled_data_comp = model.decode(bot_noised_comp).detach().cpu()
        sampled_data.append(sampled_data_comp)
    sampled_data = np.concatenate(sampled_data, axis=0)

    # sampled_data = np.clip(sampled_data, gt_min, gt_max)
    sampled_data = remove_out_of_range(sampled_data, * gt_range)
    print(f"[log] sampled data num {sampled_data.shape[0]}")
    sampled_data = calculate_sample_point_from_point_cloud(sampled_data,
                                                           min_dist_thre=0.5)
    output_file = "sampled_prop.pkl"
    with open(output_file, 'wb') as f:
        pkl.dump(sampled_data, f)
        print(f"[log] final data num {sampled_data.shape[0]} output to {output_file}")

    # drawer.add_points(list(sampled_data[:, 0]),
    #                   list(sampled_data[:, 1]),
    #                   list(sampled_data[:, 2]),
    #                   label='sampled',
    #                   color='red')
    # drawer.draw()