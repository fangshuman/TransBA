import ipdb
import torch
import torch.nn as nn

from models import make_model
from dataset import make_loader



def predict(model, data_loader):
    model.eval()
    total_num = 0
    count = 0
    preds_ls = []

    with torch.no_grad():
        for i, (inputs, labels, indexs) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            _, preds = torch.max(output.data, dim=1)
            preds_ls.append(preds)
            
            total_num += inputs.size(0)
            count += (preds == labels).sum().item()
    
    return count, total_num, preds_ls



    