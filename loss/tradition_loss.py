import torch
from .metrics import accuracy, accuracy_regression

def tradition_loss(model, x_natural, y):
    """
    Adversarial training.
    """

    tmp_y = y
    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()

    # calculate robust loss
    logits_natural = model(x_natural)
    loss_natural = loss_func(logits_natural, tmp_y)
    batch_metrics = {'clean_loss': loss_natural.item()*y.size(0), 'adv_loss': 0, 'clean_acc_num': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num': 0, 'reg':0}

    return loss_natural, batch_metrics




def tradition_loss_multi(ensemble, x_natural, y):
    loss = 0
    batch_metrics = {'clean_loss_avg': 0, 'adv_loss': 0, 'clean_acc_num_avg': 0, 'adversarial_acc_num': 0}
    for model in ensemble.models:
        tmp_loss, tmp_batch_metrics = tradition_loss(model, x_natural, y)
        loss = loss + tmp_loss
        batch_metrics['clean_loss_avg'] = batch_metrics['clean_loss_avg'] + tmp_batch_metrics['clean_loss_avg']
        batch_metrics['clean_acc_num_avg'] = batch_metrics['clean_acc_num_avg'] + tmp_batch_metrics['clean_acc_num_avg']

    batch_metrics['clean_loss_avg'] = batch_metrics['clean_loss_avg'] / len(ensemble.models)
    batch_metrics['clean_acc_num_avg'] = batch_metrics['clean_acc_num_avg'] / len(ensemble.models)
    loss = loss / len(ensemble.models)
    
    return loss, batch_metrics


def tradition_loss_multi_pdd(ensemble, x_natural, y):
    batch_metrics = {'clean_loss': 0, 'adv_loss': 0, 'clean_acc_num': 0, 'adversarial_acc_num': 0}
    loss_func = torch.nn.CrossEntropyLoss()

    
    logits_permodel = ensemble.forward_permodel(x_natural)

    logits_permodel = torch.split(logits_permodel, int(logits_permodel.shape[1]/len(ensemble.models)), 1)

    acc_number = 0
    loss_natural = 0
    for idx in range(len(ensemble.models)):
        logits_natural = logits_permodel[idx]
        loss_natural += loss_func(logits_natural, y)
        acc_number += accuracy(y, logits_natural.detach())*float(y.size(0))
    loss_natural = loss_natural/len(ensemble.models)
    acc_number = acc_number/len(ensemble.models)


    batch_metrics['clean_loss'] = loss_natural.item()
    batch_metrics['clean_acc_num'] = acc_number

    return loss_natural, batch_metrics





def tradition_loss_regression(model, x_natural, y):
    """
    Adversarial training.
    """
    tmp_y = y
    loss_func = torch.nn.MSELoss()

    pred_y = model(x_natural)
    loss_natural = loss_func(pred_y.flatten(), tmp_y)
    batch_metrics = {'clean_loss': loss_natural.item()*y.size(0), 'adv_loss': 0, 'clean_acc_num': accuracy_regression(y, pred_y.detach())*float(y.size(0)), 'adv_acc_num': 0, 'reg':0}
    
    return loss_natural, batch_metrics


def tradition_loss_classification(model, x_natural, y):
    tmp_y = y
    # define KL-loss
    loss_func = torch.nn.CrossEntropyLoss()

    # calculate robust loss
    pred_y = model(x_natural)
    loss_natural = loss_func(pred_y, tmp_y)
    batch_metrics = {'clean_loss': loss_natural.item()*y.size(0), 'adv_loss': 0, 'clean_acc_num': accuracy(y, pred_y.detach())*float(y.size(0)), 'adv_acc_num': 0, 'reg':0}
    
    return loss_natural, batch_metrics


def two_class_cross_entropy(logit_y, targets):
    pos_y = 1/(1+torch.exp(-logit_y))
    neg_y = 1 - pos_y

    cro_entro = -((pos_y.flatten()*torch.log(targets+1e-4)).sum() + (neg_y.flatten()*torch.log(1-targets+1e-4)).sum())

    # breakpoint()
    return cro_entro/targets.shape[0]


def tradition_loss_two_class_classification(model, x_natural, y):
    tmp_y = y
    loss_func = two_class_cross_entropy

    logit_y = model(x_natural)
    loss_natural = loss_func(logit_y, tmp_y)
    batch_metrics = {'clean_loss': loss_natural.item()*y.size(0), 'adv_loss': 0, 'clean_acc_num': accuracy_regression(2*y-1, logit_y.flatten().detach())*float(y.size(0)), 'adv_acc_num': 0, 'reg':0}
    
    return loss_natural, batch_metrics



def tradition_loss_multi_regression(ensemble, x_natural, y):
    loss = 0
    batch_metrics = {'clean_loss_avg': 0, 'adv_loss': 0, 'clean_acc_num_avg': 0, 'adversarial_acc_num': 0}
    for model in ensemble.models:
        tmp_loss, tmp_batch_metrics = tradition_loss_regression(model, x_natural, y)
        loss = loss + tmp_loss
        batch_metrics['clean_loss_avg'] = batch_metrics['clean_loss_avg'] + tmp_batch_metrics['clean_loss_avg']
        batch_metrics['clean_acc_num_avg'] = batch_metrics['clean_acc_num_avg'] + tmp_batch_metrics['clean_acc_num_avg']

    batch_metrics['clean_loss_avg'] = batch_metrics['clean_loss_avg'] / len(ensemble.models)
    batch_metrics['clean_acc_num_avg'] = batch_metrics['clean_acc_num_avg'] / len(ensemble.models)
    loss = loss / len(ensemble.models)
    
    return loss, batch_metrics