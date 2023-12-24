import torch
import warnings
from Model import BVTS_BiGRU_CNN, meanAlgor
from funtions import run
from processData import LinuxData_L, WindowsData_L, WholeData_L

warnings.filterwarnings("ignore")
cuda = 'cuda:1'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

acc = 0
recall = 0
precision = 0
F1test = 0
AUCtest = 0

Lengths = [30, 50, 70, 90, 110, 130, 150, 170]

for Length in Lengths:
    print("When length is " + str(Length))
    train_loader, val_loader, test_loader = LinuxData_L(Length)
    res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
    for i in range(10):
        print('Linux第{0}次：'.format(i+1))
        model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
        acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
        res['Acc'].append(acc)
        res['Rec'].append(recall)
        res['Pre'].append(precision)
        res['F1'].append(F1test)
        res['AUC'].append(AUCtest)
    print("Linux: ")
    print(meanAlgor(res))

    train_loader, val_loader, test_loader = WindowsData_L(Length)
    res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
    for i in range(10):
        print('Linux第{0}次：'.format(i+1))
        model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
        acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
        res['Acc'].append(acc)
        res['Rec'].append(recall)
        res['Pre'].append(precision)
        res['F1'].append(F1test)
        res['AUC'].append(AUCtest)
    print("Windows: ")
    print(meanAlgor(res))

    train_loader, val_loader, test_loader = WholeData_L(Length)
    res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
    for i in range(10):
        print('Linux第{0}次：'.format(i+1))
        model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
        acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
        res['Acc'].append(acc)
        res['Rec'].append(recall)
        res['Pre'].append(precision)
        res['F1'].append(F1test)
        res['AUC'].append(AUCtest)
    print("Whole: ")
    print(meanAlgor(res))