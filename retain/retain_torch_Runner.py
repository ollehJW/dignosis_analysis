import numpy as np
import time
from retain_torch.Retain_torch import Retain
import torch
import torch.nn as nn
from torch.optim import Adadelta
import math
from operator import itemgetter
from utils import metric_report
import json
import pickle
import os

def train_predict(config):

    encodeddxs_dir = config['parameters']['encodeddxs_dir']
    train_size = config['parameters']['train_size']
    epochs = config['parameters']['epochs']
    batch_size = config['parameters']['batch_size']
    learning_rate = config['parameters']['learning_rate']
    diagnosis_size = config['parameters']['diagnosis_size']
    L2_decay = config['parameters']['L2_decay']
    best_metric = config['parameters']['best_metric']
    topk = config['parameters']['topk']
    save_best_model = config['parameters']['save_best_model']
    os.makedirs(save_best_model, exist_ok=True)

    with open(config['parameters']['encodeddxs_dir'], 'rb') as f:
        patients = pickle.load(f)

    patients_num = len(patients)
    train_patient_num = int(patients_num * config['parameters']['train_size'])
    patients_train = patients[0:train_patient_num]
    test_patient_num = patients_num - train_patient_num
    patients_test = patients[train_patient_num:]

    train_batch_num = int(np.ceil(float(train_patient_num) / batch_size))
    test_batch_num = int(np.ceil(float(test_patient_num) / batch_size))

    model = Retain(inputDimSize=diagnosis_size, embDimSize=300, alphaHiddenDimSize=200, betaHiddenDimSize=200, outputDimSize=diagnosis_size)

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("Structure of the layer: " + str(list(i.size())))
        for j in i.size():
            l *= j
        print("Layer parameters: " + str(l))
        k = k + l
    print("The total number of parameters: " + str(k))

    optimizer = Adadelta(model.parameters(), lr=learning_rate, weight_decay=L2_decay)
    loss_mce = nn.BCELoss(reduction='sum')
    model = model.cuda(device=0)

    best_metric_value = 0

    for epoch in range(epochs):
        starttime = time.time()
    
        # Train Process
        model.train()
        all_loss = 0.0
        for batch_index in range(train_batch_num):
            patients_batch = patients_train[batch_index * batch_size:(batch_index + 1) * batch_size]
            patients_batch_reshape, patients_lengths = model.padTrainMatrix(patients_batch)  # maxlen × n_samples × inputDimSize
            batch_x = patients_batch_reshape[0:-1]  # Get the first n-1 as x, to predict the value of the next n-1 days
            batch_y = patients_batch_reshape[1:, :, :diagnosis_size]
            optimizer.zero_grad()
            # h0 = model.initHidden(batch_x.shape[1])
            batch_x = torch.tensor(batch_x, device=torch.device('cuda:0'))
            batch_y = torch.tensor(batch_y, device=torch.device('cuda:0'))
            y_hat = model(batch_x)
            mask = out_mask2(y_hat, patients_lengths)
            # By mask, set the network output outside the corresponding sequence length to 0
            y_hat = y_hat.mul(mask)
            batch_y = batch_y.mul(mask)
            # (seq_len, batch_size, out_dim)->(seq_len*batch_size*out_dim, 1)->(seq_len*batch_size*out_dim, )
            y_hat = y_hat.view(-1, 1).squeeze()
            batch_y = batch_y.view(-1, 1).squeeze()

            loss = loss_mce(y_hat, batch_y)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        print("Train:Epoch-" + str(epoch) + ":" + str(all_loss) + " Train Time:" + str(time.time() - starttime))

        # Evaluation Process
        model.eval()
        NDCG = 0.0
        RECALL = 0.0
        DAYNUM = 0.0
        all_loss = 0.0
        gbert_pred = []
        gbert_true = []
        gbert_len = []

        for batch_index in range(test_batch_num):
            patients_batch = patients_test[batch_index * batch_size:(batch_index + 1) * batch_size]
            patients_batch_reshape, patients_lengths = model.padTrainMatrix(patients_batch)
            batch_x = patients_batch_reshape[0:-1]
            batch_y = patients_batch_reshape[1:, :, :diagnosis_size]
            batch_x = torch.tensor(batch_x, device=torch.device('cuda:0'))
            batch_y = torch.tensor(batch_y, device=torch.device('cuda:0'))
            y_hat = model(batch_x)
            mask = out_mask2(y_hat, patients_lengths)
            loss = loss_mce(y_hat.mul(mask), batch_y.mul(mask))

            all_loss += loss.item()
            y_hat = y_hat.detach().cpu().numpy()
            ndcg, recall, daynum = validation(y_hat, patients_batch, patients_lengths, topk)
            NDCG += ndcg
            RECALL += recall
            DAYNUM += daynum
            gbert_pred.append(y_hat)
            gbert_true.append(batch_y.cpu())
            gbert_len.append(patients_lengths)

        avg_NDCG = NDCG / DAYNUM
        avg_RECALL = RECALL / DAYNUM
        y_pred_all, y_true_all = batch_squeeze(gbert_pred, gbert_true, gbert_len)
        acc_container = metric_report(y_pred_all, y_true_all, 0.2)
        print("Test:Epoch-" + str(epoch) + " Loss:" + str(all_loss) + " Test Time:" + str(time.time() - starttime))
        print("Test:Epoch-" + str(epoch) + " NDCG:" + str(avg_NDCG) + " RECALL:" + str(avg_RECALL))
        print("Test:Epoch-" + str(epoch) + " Jaccard:" + str(acc_container['jaccard']) +
                " f1:" + str(acc_container['f1']) + " prauc:" + str(acc_container['prauc']) + " roauc:" + str(
                acc_container['auc']))
    
        if best_metric_value < acc_container[best_metric]:
            best_acc_container = acc_container
            best_metric_value = acc_container[best_metric]
            torch.save(model, save_best_model + '/best_model.pt')
            print("Save Best Model!!!")

        print("")

    print("Best model: " + "Jaccard:" + str(best_acc_container['jaccard']) +
                " f1:" + str(best_acc_container['f1']) + " prauc:" + str(best_acc_container['prauc']) + " roauc:" + str(
                best_acc_container['auc']))


def out_mask2(out, data_len, device=torch.device('cuda:0')):
    """
    Parameters
    :param out: network output
    :param data_len: length list
    :param device: gpu
    :return: mask matrix
    """
    mask = torch.zeros_like(out, device=device)
    # Determine 123 or 213
    if mask.size(0) == len(data_len):
        for i, l in enumerate(data_len):
            # 对应位置长度-1
            mask[i, :l-1] = 1
    else:
        for i, l in enumerate(data_len):
            # 对应位置长度-1
            mask[:l-1, i] = 1
    return mask


def batch_squeeze(gbert_pred, gbert_true, gbert_len):
    """
    Flatten the network output of multi-label classification and the ground truth into a binary classification format

    :param gbert_pred: network output: shape=(seq_len, batch_size, out_dim)
    :param gbert_true: true label: shape=(seq_len, batch_size, out_dim)
    :param gbert_len: sequence length: list
    :return: ndarray, ndarray, shape=(seq_len*batch_size*out_dim, 1)
    """
    y_pred_all = []
    y_true_all = []
    for b in range(len(gbert_len)):
        y_pred = np.transpose(gbert_pred[b], (1, 0, 2))
        y_true = np.transpose(gbert_true[b], (1, 0, 2))
        v_len = gbert_len[b]
        for p in range(y_pred.shape[0]):
            for v in range(v_len[p] - 1):
                y_pred_all.append(y_pred[p][v].reshape(1, -1))  # shape:(283, )->(1, 283)
                y_true_all.append(y_true[p][v].reshape(1, -1))  # shape:(283, )->(1, 283)
    y_pred_all = np.concatenate(y_pred_all)     # shape=(seq_len*batch_size*out_dim, 1)
    y_true_all = np.concatenate(y_true_all)     # shape=(seq_len*batch_size*out_dim, 1)
    return y_pred_all, y_true_all


# Calculate the topk according to the training y_hat and y
# Here y_true is the actual input without padTrainMatrix
def validation(y_hat, y_true, length, topk):
    # Change the dimension to maxlen × n_samples × outputDimSize
    y_hat = np.transpose(y_hat, (1, 0, 2))

    NDCG = 0.0
    RECALL = 0.0
    daynum = 0

    n_patients = y_hat.shape[0]
    for i in range(n_patients):
        predict_one = y_hat[i]
        y_true_one = y_true[i]
        len_one = length[i]

        # Subtract 1 because the prediction is for the 2nd~nth day, excluding the first day
        for i in range(len_one - 1):
            y_pred_day = predict_one[i]
            y_true_day = y_true_one[i + 1]
            ndcg, lyt, ltp = evaluate_predict_performance(y_pred_day.flatten(), y_true_day, topk)
            NDCG += ndcg
            recall = 0.0
            if lyt != 0:
                recall += ltp * 1.0 / lyt
            else:
                recall += 1.0
            RECALL += recall
            daynum += 1

    return NDCG, RECALL, daynum


# Calculate topk for each day
def evaluate_predict_performance(y_pred, y_bow_true, topk=30):
    sorted_idx_y_pred = np.argsort(-y_pred)

    if topk == 0:
        sorted_idx_y_pred_topk = sorted_idx_y_pred[:len(y_bow_true)]
    else:
        sorted_idx_y_pred_topk = sorted_idx_y_pred[:topk]

    sorted_idx_y_true = y_bow_true

    true_part = set(sorted_idx_y_true).intersection(set(sorted_idx_y_pred_topk))  # Coincidence part, used to calculate ndcg
    idealDCG = 0.0
    for i in range(len(sorted_idx_y_true)):
        idealDCG += (2 ** 1 - 1) / math.log(1 + i + 1)

    DCG = 0.0
    for i in range(len(sorted_idx_y_true)):
        if sorted_idx_y_true[i] in true_part:
            DCG += (2 ** 1 - 1) / math.log(1 + i + 1)

    # print('true lab size: %d, intersection part size: %d' %(len(sorted_idx_y_true), len(true_part)))
    if idealDCG != 0:
        NDCG = DCG / idealDCG
    else:
        NDCG = 1
    # print('NDCG: ' + str(NDCG))
    return NDCG, len(sorted_idx_y_true), len(true_part)

if __name__ == "__main__":
    with open("/home/jongwook95.lee/study/dignosis_analysis/retain/parameters.params", 'r') as cfg:
        config = json.load(cfg)
    train_predict(config)
