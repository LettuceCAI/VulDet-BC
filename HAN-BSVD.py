import warnings
from Model import HAN_BSVD
import torch
from funtions import run, meanAlgor
from processData import LinuxData_HANBSVD, WindowsData_HANBSVD, WholeData_HANBSVD

warnings.filterwarnings("ignore")
cuda = 'cuda:1'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

acc = 0
recall = 0
precision = 0
F1test = 0
AUCtest = 0

train_loader, val_loader, test_loader, weight = LinuxData_HANBSVD()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = HAN_BSVD(weights=weight)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print(meanAlgor(res))
print('----------------------------------------------------------')

train_loader, val_loader, test_loader, weight = WindowsData_HANBSVD()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Window第{0}次：'.format(i+1))
    model = HAN_BSVD(weights=weight)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print(meanAlgor(res))
print('----------------------------------------------------------')

train_loader, val_loader, test_loader, weight = WholeData_HANBSVD()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Whole第{0}次：'.format(i+1))
    model = HAN_BSVD(weights=weight)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print(meanAlgor(res))