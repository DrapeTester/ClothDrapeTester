import os
import os.path as osp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from proputil import cvtLinearGUIToSI, cvtNonlinearGUIToSI, cvtLinearSIToGUI, cvtNonlinearSIToGUI
from resnet import ResNet18Model


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


only_evaluate_first = False


def load_png_data_from(datadir):
    num_views = 4
    data = []
    import glob
    files = [
        i for i in glob.glob(f"{datadir}/*png")
        if i.find("mask") == -1 and i.find("color") == -1
    ]
    files = sorted(files)
    # print(f"load files {files}")
    assert len(files) == num_views
    from matplotlib import pyplot as plt
    from PIL import Image
    for i in range(num_views):

        cont = np.array(Image.open(files[i]).convert('L'))
        if np.max(cont) > 250:
            cont[cont > 250] = 0
        # cont = remove_14mm(cont)
        # print(cont.shape)
        plt.subplot(1, 4, i + 1)
        data.append(cont)
        plt.imshow(cont)
    # plt.show()
    # exit()
    data = np.array(data)
    feature = None
    if True:
        feature_json = osp.join(datadir, "feature.json")
        import json
        if osp.exists(feature_json) == True:
            with open(feature_json, 'r') as f:
                cont = json.load(f)
                feature = cont[3:]
    return data, feature


# 2. predict
def predict(data):
    data = torch.from_numpy(data).float().to(device)[None, :, :, :]
    # print(data.shape)

    pred = model(data).cpu().detach().numpy()
    # swap
    # print(pred)
    for j in range(pred.shape[0]):
        if pred[j][1] > pred[j][0]:
            tmp = pred[j][0]
            pred[j][0] = pred[j][1]
            pred[j][1] = tmp
            if len(pred[j]) > 3:
                tmp = pred[j][3]
                pred[j][3] = pred[j][4]
                pred[j][4] = tmp
    return pred


def inference(model, data):
    assert data.shape == (4, 180, 240)
    predLst = []
    for direction in [1, -1]:
        for i in range(4):
            data = np.roll(data, i, axis=0)[::direction, :, :].copy()
            pred = predict(data)
            predLst.append(pred)

    predLst = np.squeeze(np.array(predLst))
    # predMean = np.mean(predLst, axis=0)
    # predStd = np.std(predLst, axis=0)
    # predMin = np.min(predLst, axis=0)
    # predMax = np.max(predLst, axis=0)
    return predLst
    # print(f"pred mean {predMean} pred std {predStd}; pred min {predMin} max {predMax}")


def getDensity(dataDir):
    path = osp.join(dataDir, "density.txt")
    with open(path, 'r') as f:
        name, densityStr = f.readlines()[0].split(" ")
        # assert dataDir.find(
        #     name
        # ) != -1, f"please check the correspondence between {dataDir} and {name}"

        try:
            density = float(densityStr)
        except Exception as e:
            print(f"fail to convert {densityStr} to number, from {path}")
            exit(1)
    assert 10 < density < 1000, f"density {density}[g] is illegal"
    return density  # unit: g


def convertGUIToSI(gui):
    if len(gui.shape) == 1:
        gui = np.expand_dims(gui, 0)
    assert len(gui.shape) == 2
    if gui.shape[1] == 3:
        si = cvtLinearGUIToSI(gui)
    elif gui.shape[1] == 6:
        si = cvtNonlinearGUIToSI(gui)
    return si


def convertSIToGUI(si):
    if len(si.shape) == 1:
        si = np.expand_dims(si, 0)
    assert len(si.shape) == 2
    if si.shape[1] == 3:
        gui = cvtLinearSIToGUI(si)
    elif si.shape[1] == 6:
        gui = cvtNonlinearSIToGUI(si)
    return gui


def Unnormalize(normalizedGUI, density):
    assert 0 <= (normalizedGUI).all() <= 100
    normalizedensity = 300
    # 1. convert to normalized SI
    ratio = density / normalizedensity
    normalizedSI = convertGUIToSI(normalizedGUI)

    unnormalizedSI = normalizedSI * ratio

    unnormalizedGUI = convertSIToGUI(unnormalizedSI)
    return unnormalizedGUI


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    np.set_printoptions(suppress=True, precision=1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    model_path = args.model_path
    data_dir = args.data_dir

    assert os.path.exists(model_path)  # model_path must be a valid path
    assert os.path.exists(data_dir)  # data_dir must be a valid path

    # 1. load the network
    device = torch.device("cpu")
    model = torch.load(model_path).to(device)
    print(f"[info] load model from {model_path}")
    model.eval()
    print(model.output_mean)
    print(f"[info] load from real data {data_dir}")

    # 2. check all sub datadirs
    data_data_lst = [
        os.path.join(data_dir, i) for i in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, i))
    ]
    for data_dir in data_data_lst:

        # 1. network inference
        data, feature = load_png_data_from(data_dir)
        if data.shape == (4, 360, 480):
            data = data[:, ::2, ::2]

        basename = os.path.basename(data_dir)

        # 2. unnoramlized
        density = getDensity(data_dir)
        print(f"-----evaluation for {basename} density {density}-----")
        NormalizedGUI = inference(model, data)
        NormalizedGUIMean = np.mean(NormalizedGUI, axis=0)

        UnnormalizedGUI = Unnormalize(NormalizedGUI, density)

        GUIMean = np.mean(UnnormalizedGUI, axis=0)
        SI = np.round(convertGUIToSI(GUIMean), 1)
        print(f"bending stiffness: {SI}")
