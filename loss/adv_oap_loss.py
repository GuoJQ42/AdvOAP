import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import random
from .metrics import accuracy
from attacks.pgd import PGD_attack, PGD_attack_pro, PGD_attack_two_class_classification

def adv_oap(ensemble_model, x_natural, y, device, step_size=0.002, epsilon=0.031, perturb_steps=40, attack_type='l_infty', alpha = 2, beta = 1, num_classes = 10):
    """
    The number of base learners should be equal to the number of classes minus 1.
    """
    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()

    final_loss = 0

    batch_metrics = {}

    x_advs = []
    
    for model in ensemble_model.models:
        x_adv = PGD_attack(model, x_natural, y, epsilon, perturb_steps, step_size, attack_type, bias=None)
        x_advs.append(x_adv)
    
    preds_exclude_y = []
    y_one_hot = F.one_hot(y, num_classes = num_classes)
    y_one_hot_rev = 1 - y_one_hot
    y_one_hot_rev = y_one_hot_rev.bool()
    

    for idx, model in enumerate(ensemble_model.models):
        model.train()
        logits_adv = model(x_advs[idx])

        loss_robust = loss_func(logits_adv, y)
        pred = F.softmax(logits_adv, dim = 1) 

        pred_non_y = torch.masked_select(pred, y_one_hot_rev).view([pred.shape[0],-1])
        preds_exclude_y.append(pred_non_y)
        
        logits_natural = model(x_natural)
        loss_natural = loss_func(logits_natural, y)

        final_loss += loss_robust

        if len(batch_metrics) == 0:
            batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item()*float(y.size(0)), 'clean_acc_num': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num': accuracy(y, logits_adv.detach())*float(y.size(0))}
        else:
            tmp_batch_metrics = {'clean_loss': loss_natural.item(), 'adv_loss': loss_robust.item()*float(y.size(0)), 'clean_acc_num': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num': accuracy(y, logits_adv.detach())*float(y.size(0))}
            for key in batch_metrics.keys():
                batch_metrics[key] = batch_metrics[key] + tmp_batch_metrics[key]
        
    
    for key in batch_metrics.keys():
        batch_metrics[key] = batch_metrics[key] / len(ensemble_model.models)
    
    preds_exclude_y = torch.stack(preds_exclude_y, dim=1)
    preds_exclude_y_normalized = F.normalize(preds_exclude_y, p=1, dim=2)

    preds_exclude_y_normalized_mean = preds_exclude_y_normalized.mean(dim=1)

    entropy = -((preds_exclude_y_normalized_mean+1e-6) * torch.log(preds_exclude_y_normalized_mean)).sum(dim=1).mean()

    matrix = torch.matmul(preds_exclude_y_normalized, preds_exclude_y_normalized.permute(0, 2, 1)) 
    matrix = matrix + (torch.eye(matrix.shape[1]).to(device))*1e-6

    adp_term = torch.logdet(matrix).mean()

    final_loss = final_loss - alpha*adp_term - beta*entropy
    batch_metrics['reg'] = entropy.item()*float(y.size(0))

    return final_loss, batch_metrics

