import torch
import warnings
from Model import Con_BiGRU_CNN, onlyAttention, BiGRUwithoutAttention
from funtions import run, meanAlgor
from processData import LinuxData, WindowsData, WholeData, LinuxData_Con, WindowsData_Con, WholeData_Con

warnings.filterwarnings("ignore")
cuda = 'cuda:1'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

acc = 0
recall = 0
precision = 0
F1test = 0
AUCtest = 0

# -----------------------Concatenation--------------------------
print('Concatenation:')

train_loader, val_loader, test_loader = LinuxData_Con()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = Con_BiGRU_CNN(256, 64)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tLinux: ")
print(meanAlgor(res))

train_loader, val_loader, test_loader = WindowsData_Con()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = Con_BiGRU_CNN(256, 64)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tWindows: ")
print(meanAlgor(res))

train_loader, val_loader, test_loader = WholeData_Con()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = Con_BiGRU_CNN(256, 64)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tWhole: ")
print(meanAlgor(res))

print("---------------------------------------------------------------")

# -----------------------Attention--------------------------
print('Attention:')

train_loader, val_loader, test_loader = LinuxData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = onlyAttention(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tLinux: ")
print(meanAlgor(res))

train_loader, val_loader, test_loader = WindowsData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = onlyAttention(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tWindows: ")
print(meanAlgor(res))

train_loader, val_loader, test_loader = WholeData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = onlyAttention(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tWhole: ")
print(meanAlgor(res))

print("---------------------------------------------------------------")

# -----------------------BiGRU--------------------------
print('BiGRU:')

train_loader, val_loader, test_loader = LinuxData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = BiGRUwithoutAttention(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tLinux: ")
print(meanAlgor(res))

train_loader, val_loader, test_loader = WindowsData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = BiGRUwithoutAttention(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tWindows: ")
print(meanAlgor(res))

train_loader, val_loader, test_loader = WholeData()
res = {'Acc': [], 'Rec': [], 'Pre': [], 'F1': [], 'AUC': []}
for i in range(10):
    print('Linux第{0}次：'.format(i+1))
    model = BiGRUwithoutAttention(258, 128, 256, 64, 0.7)
    acc, recall, precision, F1test, AUCtest = run(model, device, train_loader, val_loader, test_loader)
    res['Acc'].append(acc)
    res['Rec'].append(recall)
    res['Pre'].append(precision)
    res['F1'].append(F1test)
    res['AUC'].append(AUCtest)
print("\t\tWhole: ")
print(meanAlgor(res))