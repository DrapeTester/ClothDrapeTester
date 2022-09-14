import numpy as np
np.random.seed(0)

data = '''0e0 0.0
1e2 10.0
1e3 28.0
3e3 48.0
2e4 65.0
2e5 83.0
2e6 100.0'''

gui_to_SI_map = []
for line in data.split("\n"):
    SI, gui = line.split(" ")
    SI = float(SI)
    gui = float(gui)
    gui_to_SI_map.append((gui, SI))
nonlinearScale = 0.1 * 1e3 # nonlinear unit [g.mm3] = 1000 [g.mm2]


def __cvtGuiToSIScalar(gui_value):
    sign = -1 if gui_value < 0 else 1
    gui_value = np.abs(gui_value)
    global gui_to_SI_map
    for i in range(len(gui_to_SI_map) - 1):
        cur_gui = gui_to_SI_map[i][0]
        cur_SI = gui_to_SI_map[i][1]
        next_gui = gui_to_SI_map[i + 1][0]
        next_SI = gui_to_SI_map[i + 1][1]
        if gui_value >= cur_gui and gui_value < next_gui:
            this_SI = (gui_value - cur_gui) / (next_gui - cur_gui) * (
                next_SI - cur_SI) + cur_SI
            return sign * this_SI
    # gui is greater than up limit, return the max number
    return sign * next_SI


def __cvtSIToGuiScalar(SI_value):
    sign = -1 if SI_value < 0 else 1
    SI_value = np.abs(SI_value)
    for i in range(len(gui_to_SI_map) - 1):
        cur_gui = gui_to_SI_map[i][0]
        cur_SI = gui_to_SI_map[i][1]
        next_gui = gui_to_SI_map[i + 1][0]
        next_SI = gui_to_SI_map[i + 1][1]

        if SI_value >= cur_SI and SI_value < next_SI:
            this_gui = (SI_value - cur_SI) / (next_SI - cur_SI) * (
                next_gui - cur_gui) + cur_gui
            return sign * this_gui
    return sign * next_gui


# convert a GUI property vector [a, b, c] to SI
def __cvtVec(cvtFunc, oldValueVec):
    len(oldValueVec) == 3

    oldValueVec = np.squeeze(oldValueVec)
    assert len(oldValueVec.shape) == 1
    newValueVec = []
    for i in range(len(oldValueVec)):
        oldVal = oldValueVec[i]
        sign = 1 if oldVal >= 0 else -1
        newVal = cvtFunc(np.abs(oldVal)) * sign
        newValueVec.append(newVal)
    return np.array(newValueVec)


def __cvtMat(cvtFunc, oldValueMat):
    assert len(oldValueMat.shape) == 2
    newValueMat = []
    for i in range(oldValueMat.shape[0]):
        curGuiValue = oldValueMat[i]
        curSIValue = __cvtVec(cvtFunc, curGuiValue)
        newValueMat.append(curSIValue)
    newValueMat = np.array(newValueMat)
    return newValueMat

def cvtLinearGUIToSI(GUI):
    isVec = len(GUI.shape) == 1
    isMat = len(GUI.shape) == 2
    assert isVec or isMat
    if isVec:
        assert len(GUI )== 3
        return __cvtVec(__cvtGuiToSIScalar, GUI)
    elif isMat:
        assert GUI.shape[1] == 3
        return __cvtMat(__cvtGuiToSIScalar, GUI)
    
def cvtLinearSIToGUI(SI):
    isVec = len(SI.shape) == 1
    isMat = len(SI.shape) == 2
    assert isVec or isMat
    if isVec:
        return __cvtVec(__cvtSIToGuiScalar, SI)
    elif isMat:
        return __cvtMat(__cvtSIToGuiScalar, SI)
from copy import deepcopy
def cvtNonlinearGUIToSI(GUI):
    # print(f"gui {GUI}")
    isVec = len(GUI.shape) == 1
    isMat = len(GUI.shape) == 2

    if isVec:
        assert len(GUI)== 6
        # 1. get linear part
        linearGUI = GUI[:3]
        nonlinearGUI = GUI[3:]
        linearSI = cvtLinearGUIToSI(linearGUI)
        nonlinearSI = cvtLinearGUIToSI(nonlinearGUI) * nonlinearScale
        SI = deepcopy(GUI)
        SI[:3] = linearSI
        SI[3:] = nonlinearSI
    elif isMat:
        assert GUI.shape[1] == 6
        linearGUI = GUI[:, :3]
        nonlinearGUI = GUI[:, 3:]
        linearSI = cvtLinearGUIToSI(linearGUI)
        nonlinearSI = cvtLinearGUIToSI(nonlinearGUI) * nonlinearScale
        SI = deepcopy(GUI)
        SI[:, :3] = linearSI
        SI[:, 3:] = nonlinearSI
    # print(f"SI {SI}")
    # exit()
    return SI

def cvtNonlinearSIToGUI(SI):
    isVec = len(SI.shape) == 1
    isMat = len(SI.shape) == 2

    if isVec:
        assert len(SI )==6
        # 1. get linear part
        linearSI = SI[:3]
        nonlinearSI = SI[3:] / nonlinearScale
        linearGUI = cvtLinearSIToGUI(linearSI)
        nonlinearGUI = cvtLinearSIToGUI(nonlinearSI)
        GUI = deepcopy(SI)
        GUI[:3] = linearGUI
        GUI[3:] = nonlinearGUI
    
    elif isMat:
        assert SI.shape[1] == 6
        # 1. get linear part
        linearSI = SI[:, :3]
        nonlinearSI = SI[:, 3:] / nonlinearScale
        linearGUI = cvtLinearSIToGUI(linearSI)
        nonlinearGUI = cvtLinearSIToGUI(nonlinearSI)
        GUI = deepcopy(SI)
        GUI[:, :3] = linearGUI
        GUI[:, 3:] = nonlinearGUI
    return GUI

def __linearTest():
    numOfSamples = 2
    linearGUI = np.round(np.random.randn(numOfSamples, 3) * 10, 1)
    linearSI = cvtLinearGUIToSI(linearGUI)
    linearGUINew = cvtLinearSIToGUI(linearSI)
    
    print(f"linear GUI {linearGUI}")
    print(f"linear SI {linearSI}")
    print(f"linear GUI renew {linearGUINew}")
    assert (linearGUINew == linearGUI).all()
    print(f"-------test linear succ-------")
    pass

def __nonlinearTest():
    numOfSamples = 20
    nonlinearGUI = np.round((np.random.rand(numOfSamples, 6) - 0.3) * 100, 1)
    nonlinearSI =  cvtNonlinearGUIToSI(nonlinearGUI)
    nonlinearGUINew = np.round( cvtNonlinearSIToGUI(nonlinearSI), 1)
    
    print(f"nonlinear GUI {nonlinearGUI}")
    print(f"nonlinear SI {nonlinearSI}")
    print(f"nonlinear GUI renew {nonlinearGUINew}")
    assert (nonlinearGUINew == nonlinearGUI).all(), (nonlinearGUINew == nonlinearGUI)
    print(f"-------test nonlinear succ-------")
    


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # __linearTest()
    __nonlinearTest()
