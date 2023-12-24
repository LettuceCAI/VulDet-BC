import torch
import warnings
from Model import BVTS_BiGRU_CNN
from funtions import run_time
from processData import LinuxData_L, WindowsData_L, WholeData_L

warnings.filterwarnings("ignore")
cuda = 'cuda:1'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

Lengths = [30, 50, 70, 90, 110, 130, 150, 170]

print("Linux: ")
for Length in Lengths:
    print("\t\tWhen length is " + str(Length))
    train_loader, val_loader, test_loader = LinuxData_L(Length)
    model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
    ans = run_time(model, device, train_loader, val_loader, test_loader)
    print('\t\tTime:' + str(ans))

print("Windows: ")
for Length in Lengths:
    print("\t\tWhen length is " + str(Length))
    train_loader, val_loader, test_loader = WindowsData_L(Length)
    model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
    ans = run_time(model, device, train_loader, val_loader, test_loader)
    print('\t\tTime:' + str(ans))

print("Whole: ")
for Length in Lengths:
    print("\t\tWhen length is " + str(Length))
    train_loader, val_loader, test_loader = WholeData_L(Length)
    model = BVTS_BiGRU_CNN(258, 128, 256, 64, 0.7)
    ans = run_time(model, device, train_loader, val_loader, test_loader)
    print('\t\tTime:' + str(ans))