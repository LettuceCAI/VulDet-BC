import torch
import torch.utils.data as Data
from torch.utils.data import random_split
import numpy as np
import os
from gensim.models import Word2Vec

def LinuxData():
    if os.path.exists('data/Lindataset.pt'):
        dataset = torch.load('data/Lindataset.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-ubuntu', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(word)
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/Lindataset.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(45))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def LinuxData_L(Length):
    if os.path.exists('data/Length/Lin' + str(Length) + '.pt'):
        dataset = torch.load('data/Length/Lin' + str(Length) + '.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-ubuntu', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < Length:
                for j in range(Length - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:Length]

        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(int(word))
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, '../data/Length/Lin' + str(Length) + '.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(45))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def LinuxData_Con():
    if os.path.exists('data/LinCondataset.pt'):
        dataset = torch.load('data/LinCondataset.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-ubuntu', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        model = Word2Vec.load('model/word_vector_con.model')
        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = np.array([])
                for word in words:
                    senten = np.concatenate([senten, model.wv[word]])
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/LinCondataset.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(45))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def LinuxData_HANBSVD():
    if os.path.exists('data/LinWorddataset.pt'):
        dataset = torch.load('data/LinWorddataset.pt')
        weight = torch.load('data/Linweight.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-ubuntu', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        model = Word2Vec.load('model/word_vector.model')
        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(model.wv.key_to_index[word])
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        wordvoc = []
        for word in model.wv.index_to_key:
            wordvoc.append(model.wv[word])

        weight = torch.FloatTensor(wordvoc)
        torch.save(weight, 'data/Linweight.pt')
        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/LinWorddataset.pt')
    ze = torch.zeros(128)
    weight[0] = ze
    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(45))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader


def WholeData():
    if os.path.exists('data/Whodataset.pt'):
        dataset = torch.load('data/Whodataset.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '32-ubuntu', '64-windows', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(int(word))
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/Whodataset.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(107))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def WholeData_L(Length):
    if os.path.exists('data/Length/Who' + str(Length) + '.pt'):
        dataset = torch.load('data/Length/Who' + str(Length) + '.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '32-ubuntu', '64-windows', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < Length:
                for j in range(Length - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:Length]

        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(int(word))
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, '../data/Length/Who' + str(Length) + '.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(107))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def WholeData_Con():
    if os.path.exists('data/WhoCondataset.pt'):
        dataset = torch.load('data/WhoCondataset.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '32-ubuntu', '64-windows', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        model = Word2Vec.load('model/word_vector_con.model')
        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = np.array([])
                for word in words:
                    senten = np.concatenate([senten, model.wv[word]])
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/WhoCondataset.pt')
        n_train = int(len(dataset) * 0.8)
        n_val = int(len(dataset) * 0.1)
        n_test = len(dataset) - n_train - n_val
        train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(107))
        train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
        return train_loader, val_loader, test_loader

def WholeData_HANBSVD():
    if os.path.exists('data/WhoWorddataset.pt'):
        dataset = torch.load('data/WhoWorddataset.pt')
        weight = torch.load('data/Whoweight.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '32-ubuntu', '64-windows', '64-ubuntu']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        model = Word2Vec.load('model/word_vector.model')
        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(model.wv.key_to_index[word])
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        wordvoc = []
        for word in model.wv.index_to_key:
            wordvoc.append(model.wv[word])

        weight = torch.FloatTensor(wordvoc)
        torch.save(weight, 'data/Whoweight.pt')
        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/WhoWorddataset.pt')

    ze = torch.zeros(128)
    weight[0] = ze
    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(107))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def WindowsData():
    if os.path.exists('data/Windataset.pt'):
        dataset = torch.load('data/Windataset.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '64-windows']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(int(word))
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/Windataset.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(572))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def WindowsData_L(Length):
    if os.path.exists('data/Length/Win' + str(Length) + '.pt'):
        dataset = torch.load('data/Length/Win' + str(Length) + '.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '64-windows']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < Length:
                for j in range(Length - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:Length]

        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(int(word))
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, '../data/Length/Win' + str(Length) + '.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(573))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader

def WindowsData_Con():
    if os.path.exists('data/WinCondataset.pt'):
        dataset = torch.load('data/WinCondataset.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '64-windows']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        model = Word2Vec.load('model/word_vector_con.model')
        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = np.array([])
                for word in words:
                    senten = np.concatenate([senten, model.wv[word]])
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/WinCondataset.pt')

    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(572))
    train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_loader, val_loader, test_loader

def WindowsData_HANBSVD():
    if os.path.exists('data/WinWorddataset.pt'):
        dataset = torch.load('data/WinWorddataset.pt')
        weight = torch.load('data/Winweight.pt')
    else:
        X_full = []
        Y_full = np.array([])
        for arch_os in ['32-windows', '64-windows']:
            decimal_functions_path = 'dataset/vocab_opcode-' + arch_os + '.data'
            label_path = 'dataset/labels-' + arch_os + '.data'
            with open(decimal_functions_path, 'r') as f:
                X_lines = f.readlines()
            with open(label_path, 'r') as f:
                Y_lines = f.readlines()

            Y = np.array([int(number) for number in Y_lines[0].split()])
            X_full += X_lines
            Y_full = np.concatenate((Y_full, Y), axis=0)

        list_function = []
        list_code = []

        for _, assembly_code in enumerate(X_full):
            if assembly_code != '-----\n':
                assembly_code = assembly_code[:-1]
                if len(assembly_code.split('|')) == 2:
                    line = assembly_code.split('|')[1]
                else:
                    line = assembly_code.split('|')[-1]
                list_code.append(line)
            else:
                list_function.append(list_code)
                list_code = []

        list_function_afterpad = []
        for fun in list_function:
            spinnet = []
            for line in fun:
                temp = line.split(',')
                temp = [j for j in temp if j != '']
                for i in range(len(temp)):
                    temp[i] = str(int(temp[i]) + 1)
                if len(temp) < 5:
                    for k in range(5 - len(temp)):
                        temp.append('257')
                else:
                    temp = temp[0:5]
                spinnet.append(' '.join(temp))
            list_function_afterpad.append(spinnet)
        del list_function

        zeros_line = '0 0 0 0 0'
        for i in range(len(list_function_afterpad)):
            if len(list_function_afterpad[i]) < 150:
                for j in range(150 - len(list_function_afterpad[i])):
                    list_function_afterpad[i].append(zeros_line)
            else:
                list_function_afterpad[i] = list_function_afterpad[i][:150]

        model = Word2Vec.load('model/word_vector.model')
        input_data = []
        for fun_code in list_function_afterpad:
            fun_snippet = []
            for line in fun_code:
                words = line.split(' ')
                words = [j for j in words if j != '']
                senten = []
                for word in words:
                    senten.append(model.wv.key_to_index[word])
                fun_snippet.append(senten)
            input_data.append(fun_snippet)

        wordvoc = []
        for word in model.wv.index_to_key:
            wordvoc.append(model.wv[word])

        weight = torch.FloatTensor(wordvoc)
        torch.save(weight, 'data/Winweight.pt')
        input_batch = torch.tensor(input_data)
        Y_full = torch.tensor(Y_full)
        dataset = Data.TensorDataset(input_batch, Y_full)
        torch.save(dataset, 'data/WinWorddataset.pt')

    ze = torch.zeros(128)
    weight[0] = ze
    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    n_test = len(dataset) - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(572))
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader, weight