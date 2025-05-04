import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from .metrics import accuracy


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model, x_natural, y, step_size=0.003, epsilon=0.031, perturb_steps=20, beta=7, attack='linf-pgd'):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    loss_fun = nn.CrossEntropyLoss()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    p_natural = F.softmax(model(x_natural), dim=1)
    
    if attack == 'l_infty':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    

    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    loss_natural = loss_fun(logits_natural, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_natural, dim=1))
    # breakpoint()
    loss = loss_natural + beta * loss_robust
    
    batch_metrics = {'clean_loss_avg': loss_natural.item()/ batch_size, 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0))}
        
    return loss, batch_metrics









def trades_loss_multi(ensemblemodel, x_natural, y, alpha, step_size=0.003, epsilon=0.031, perturb_steps=20, beta=5, attack_type='linf-pgd'):
    """
    TRADES training (Zhang et al, 2019).
    """

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    ensemblemodel.eval()
    batch_size = len(x_natural)
    loss_fun = nn.CrossEntropyLoss()

    x_advs = []

    for model in ensemblemodel.models:

        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        p_natural = F.softmax(model(x_natural), dim=1)
    
        if attack_type == 'l_infty':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif attack_type == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
                loss.backward(retain_graph=True)
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            raise ValueError(f'Attack={attack_type} not supported for TRADES training!')
        
        model.train()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        x_advs.append(x_adv)
    

    final_loss = 0

    logits_adv_average = 0
    logits_natural_average = 0
    loss_natural_sum = 0

    for x_adv, model in zip(x_advs, ensemblemodel.models):
    
        # calculate robust loss
        logits_natural = model(x_natural)
        loss_natural = loss_fun(logits_natural, y)
        final_loss += loss_natural

        loss_natural_sum += loss_natural

        logits_adv = model(x_adv)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                        F.softmax(logits_natural, dim=1))
        # final_loss += beta * (1-alpha)* loss_robust
        final_loss += beta * loss_robust

        logits_natural_average += logits_natural
        logits_adv_average += logits_adv
    
    # logits_natural_average = logits_natural_average/len(ensemblemodel.models)
    # logits_adv_average = logits_adv_average/len(ensemblemodel.models)
    logits_natural_average = logits_natural_average
    logits_adv_average = logits_adv_average
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv_average, dim=1),
                                                        F.softmax(logits_natural_average, dim=1))
    # final_loss += beta * alpha *loss_robust
    final_loss += alpha *loss_robust

    
    batch_metrics = {'clean_loss_avg': loss_natural_sum.item()/ batch_size, 'adv_loss': loss_robust.item(), 'clean_acc_num_avg': accuracy(y, logits_natural.detach())*float(y.size(0)), 'adv_acc_num_avg': accuracy(y, logits_adv.detach())*float(y.size(0)), 'final_loss': final_loss.item()}
        
    return final_loss, batch_metrics