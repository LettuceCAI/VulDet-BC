import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import time

def train(model, train_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    train_loss = []

    model.train()
    
    for data, lable in train_loader:
        lable = lable.to(device)
        data = data.to(device)

        optimizer.zero_grad()
        
        out = model(data)
        # loss = criterion(out, lable.reshape(-1, 1))
        loss = criterion(out, lable.long())
        train_loss.append(loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()  
    return np.mean(train_loss)

def test(model, loader, device):
    model.eval()
    
    correct = 0
    total = 0
    target_num = torch.zeros((1, 2))
    predict_num = torch.zeros((1, 2))
    acc_num = torch.zeros((1, 2))
    prob_all = []
    lable_all = []
    for data, lable in loader:
        lable = lable.to(device)
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += lable.size(0)
        correct += predicted.eq(lable.data).cpu().sum()
        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, lable.long().data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)
        prob_all.extend(predicted.cpu().numpy())
        lable_all.extend(lable.cpu().numpy())
    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)
    
    recall = (recall.numpy()[0]*100).round(7)
    precision = (precision.numpy()[0]*100).round(7)
    F1 = (F1.numpy()[0]*100).round(7)
    accuracy = (accuracy.numpy()[0]*100).round(7)
    auc = (roc_auc_score(lable_all,prob_all)*100).round(7)
    return accuracy, recall, precision, F1, auc

def meanAlgor(res):
    ans = {'Acc': 0, 'Rec': 0, 'Pre': 0, 'F1': 0, 'AUC': 0}
    for k, v in res.items():
        temp = np.array(res[k])
        ans[k] = np.mean(temp)

    return ans

def run(model, device, train_loader, val_loader, test_loader):
    model = model.to(device)
    acc = 0
    recall = 0
    precision = 0
    F1test = 0
    AUCtest = 0
    max = 0
    for epoch in range(1, 51):
        loss = train(model, train_loader, device)
        Trainaccuracy, Trainrecall, Trainprecision, TrainF1, TrainAUC = test(model, train_loader, device)
        Valaccuracy, Valrecall, Valprecision, ValtF1, ValAUC = test(model, val_loader, device)
        Testaccuracy, Testrecall, Testprecision, TestF1, TestAUC = test(model, test_loader, device)
        avg = (Valaccuracy + Valrecall[1] + Valprecision[1] + ValtF1[1] + ValAUC) / 5
        if max < avg:
            max = avg
            acc = Testaccuracy
            recall = Testrecall
            precision = Testprecision
            F1test = TestF1
            AUCtest = TestAUC

    print('When the verification set is the best, the result of the test set is:')
    print(f'Test Acc: {acc}, Test recall: {recall[1]}, Test precision: {precision[1]}, Test F1: {F1test[1]}, Test AUC: {AUCtest}')
    del model
    return acc, recall[1], precision[1], F1test[1], AUCtest

def run_time(model, device, train_loader, val_loader, test_loader):
    time_begin = time.time()
    model = model.to(device)
    acc = 0
    recall = 0
    precision = 0
    F1test = 0
    AUCtest = 0
    max = 0
    for epoch in range(1, 51):
        loss = train(model, train_loader)
        Trainaccuracy, Trainrecall, Trainprecision, TrainF1, TrainAUC = test(model, train_loader)
        Valaccuracy, Valrecall, Valprecision, ValtF1, ValAUC = test(model, val_loader)
        Testaccuracy, Testrecall, Testprecision, TestF1, TestAUC = test(model, test_loader)
        avg = (Valaccuracy + Valrecall[1] + Valprecision[1] + ValtF1[1] + ValAUC) / 5
        # avg = (Valaccuracy + Valrecall[0] + Valprecision[0] + ValtF1[0] + ValAUC) / 5
        if max < avg:
            max = avg
            acc = Testaccuracy
            recall = Testrecall
            precision = Testprecision
            F1test = TestF1
            AUCtest = TestAUC
    time_end = time.time()

    return time_end - time_begin