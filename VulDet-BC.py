import warnings
from Model import BVTS_BiGRU_CNN
import torch
from funtions import run, meanAlgor
from processData import LinuxData, WindowsData, WholeData

warnings.filterwarnings("ignore")
cuda = 'cuda:1'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

acc = 0
recall = 0
precision = 0
F1test = 0
AUCtest = 0

train_loader, val_loader, test_loader = LinuxData()
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
print(meanAlgor(res))

print('----------------------------------------------------------')

train_loader, val_loader, test_loader = WindowsData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Windows第{0}次：'.format(i+1))
    model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print(meanAlgor(res))

print('----------------------------------------------------------')

train_loader, val_loader, test_loader = WholeData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Whole第{0}次：'.format(i+1))
    model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print(meanAlgor(res))