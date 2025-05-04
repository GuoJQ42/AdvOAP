import torch
import torch.nn.functional as F 
import torch.nn as nn
import math
import torchvision.transforms as transforms

def rand_init_delta(delta, ori_data, type, eps, clip_min = 0, clip_max = 1, bias = None):
    
    if type == 'l_infty':
        delta.data = torch.zeros_like(ori_data).uniform_(-eps, eps)
    elif type == 'l_2':
        delta.data = torch.randn_like(ori_data)
        norms = torch.norm(delta.data.view([delta.data.shape[0], -1]), dim=1).view([delta.data.shape[0],1,1,1])
        # breakpoint()
        delta.data = delta.data / norms
        # Randomly scale the points to be inside the unit hypersphere
        scales = torch.rand(ori_data.shape[0]).to(delta.device) ** (1 / ori_data.shape[1]*ori_data.shape[2]*ori_data.shape[3])
        scales = scales.view([delta.data.shape[0],1,1,1])
        delta.data = delta.data * scales
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)
    if bias != None:
        delta.data = delta.data/30 + bias
    delta.data = torch.clamp(ori_data + delta.data, min=clip_min, max=clip_max) - ori_data.data
    return


def get_output_scale(output):
    std_max_out = []
    maxk = max((10,))
    # topk from big to small
    # print(output.shape)
    pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)

    scale_list = (pred_val_out[:, 0] - pred_val_out[:, 1]).reshape([output.shape[0],-1])

    return scale_list



def margin_loss(output, target):
    y_one_hot = F.one_hot(target, num_classes = 10)
    y_one_hot_rev = 1 - y_one_hot
    y_one_hot = y_one_hot.bool()
    y_one_hot_rev = y_one_hot_rev.bool()
    output_y = torch.masked_select(output, y_one_hot).view([output.shape[0], -1])
    output_non_y = torch.masked_select(output, y_one_hot_rev).view([output.shape[0], -1])
    output_non_y_max = output_non_y.max(dim=1)[0]
    # breakpoint()
    margin = (output_non_y_max.flatten() - output_y.flatten()).mean()

    return margin


"""
Margin weighted PGD for test
"""
def PGD_attack_pro(model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='l_infty', bias=None):
    model.eval()

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)
    eta_old = torch.clone(eta)

    rand_init_delta(eta, ori_data, type, per_size, clip_min = 0, clip_max = 1, bias = bias)

    eta.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()

    
    step_size_begin = per_size*2

    for i in range(num_iter):

        with torch.no_grad():
            acc = model(data + eta).max(1)[1] == target
        
        if acc.sum() == 0:
            break
        
        # ind_to_fool = acc.nonzero().flatten()
        # breakpoint()
        ind_to_fool = acc.nonzero().flatten()

        step_size = step_size_begin * math.cos(i / num_iter * math.pi * 0.5)
        # output = model(data + eta)
        output = model(data + eta)[ind_to_fool]
        scale_output = get_output_scale(output.clone().detach())
        # print(scale_output.flatten())
        # breakpoint()
        output = output/scale_output
        test_loss = loss_fn(output, target[ind_to_fool])


        # test_loss = loss_fn(output, target)
        test_loss.backward()
        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()

            eta_old.data = eta.data
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            # momutem
            a = 0.75 if i > 0 else 1.0
            eta.data = a*eta.data + (1-a)*eta_old.data
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        eta.grad.data.zero_()

    
    return torch.clamp(ori_data+eta.data, min=0, max=1)


"""
Trivial PGD
"""
def PGD_attack(model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='l_infty', bias=None):
    model.eval()
    # import random
    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    rand_init_delta(eta, ori_data, type, per_size, clip_min = 0, clip_max = 1, bias = bias)


    eta.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()

    import random

    for i in range(num_iter):
        
        output = model(data + eta)

        test_loss = loss_fn(output, target)
        test_loss.backward()
        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        eta.grad.data.zero_()
    
    return torch.clamp(ori_data+eta.data, min=0, max=1)










def PGD_attack_EOT(model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='l_infty', bias=None):
    model.eval()

    # 定义旋转变换（旋转角度范围 -30 到 30 度）
    transform = transforms.RandomRotation(degrees=(-30, 30))




    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    rand_init_delta(eta, ori_data, type, per_size, clip_min = 0, clip_max = 1, bias = bias)


    eta.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()


    for i in range(num_iter):
        
        test_loss = 0

        for _ in range(20):
            rotated_image = transform(data + eta)
            output = model(rotated_image)
            test_loss += loss_fn(output, target)
   
        test_loss.backward()
        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        eta.grad.data.zero_()
    
    return torch.clamp(ori_data+eta.data, min=0, max=1)



def two_class_cross_entropy(logit_y, targets):
    pos_y = 1/(1+torch.exp(-logit_y))
    neg_y = 1 - pos_y

    cro_entro = -((pos_y.flatten()*torch.log(targets+1e-4)).sum() + (neg_y.flatten()*torch.log(1-targets+1e-4)).sum())

    return cro_entro/targets.shape[0]




"""
Trivial PGD
"""
def PGD_attack_two_class_classification(model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='l_infty', bias=None):
    model.eval()

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    rand_init_delta(eta, ori_data, type, per_size, clip_min = 0, clip_max = 1, bias = bias)


    eta.requires_grad = True

    loss_fn = two_class_cross_entropy


    for i in range(num_iter):
        output = model(data + eta)
        test_loss = loss_fn(output, target)
        test_loss.backward()
        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        eta.grad.data.zero_()
    
    return torch.clamp(ori_data+eta.data, min=0, max=1)



def PGD_attack_with_mask(model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='l_infty', bias=None):
    model.eval()

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    rand_init_delta(eta, ori_data, type, per_size, clip_min = 0, clip_max = 1, bias = bias)


    eta.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()


    for i in range(num_iter):
        output = model(data + eta)
        test_loss = loss_fn(output, target)
        test_loss.backward()
        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        eta.grad.data.zero_()
    
    x_adv = torch.clamp(ori_data+eta.data, min=0, max=1)
    acc_mask = (model(x_adv).max(1)[1] == target).int()

    return x_adv, acc_mask



def FGSM_attack(model, data, target, per_size=8/255, type='l_infty'):
    model.eval()

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    eta.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()


    output = model(data + eta)
    test_loss = loss_fn(output, target)
    test_loss.backward()

    if type == 'l_2':
        grad = eta.grad
        grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
        grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
        grad = grad/grad_norm.view([batch_size, 1, 1, 1])
        eta.data = eta.data + per_size*grad

        eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
        factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
        factor = factor.view([batch_size, 1, 1, 1])
        eta.data = factor * eta.data

        eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

    elif type == 'l_infty':
        grad_sign = eta.grad.data.sign()
        eta.data = eta.data + per_size*grad_sign
        eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
        eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)
    eta.grad.data.zero_()
    
    return torch.clamp(ori_data+eta.data, min=0, max=1)








def PGD_attack_regression(model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='l_infty', bias=None):
    model.eval()

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    rand_init_delta(eta, ori_data, type, per_size, clip_min = 0, clip_max = 1, bias = bias)

    eta.requires_grad = True

    loss_fn = torch.nn.MSELoss()

    for i in range(num_iter):
        output = model(data + eta)
        test_loss = loss_fn(output.flatten(), target)
        test_loss.backward()
        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        eta.grad.data.zero_()
    
    return torch.clamp(ori_data+eta.data, min=0, max=1)


def PGD_attack_average_model(ensemble_model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='linfty'):
    ensemble_model.eval()

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    rand_init_delta(eta, data, type, per_size, 0, 1, None)

    eta.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(num_iter):
        test_loss = 0
        for model in ensemble_model.models:
            output = model(data + eta)
            test_loss = test_loss + loss_fn(output, target)
        test_loss = test_loss/len(ensemble_model.models)
        test_loss.backward()

        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)

        eta.grad.data.zero_()
    return eta.data



def two_stage_PGD(ensemble_model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='linfty'):
    ensemble_model.eval()

    bias_data = PGD_attack_average_model(ensemble_model, data, target, per_size, num_iter//2, step_size, type)

    x_advs = []
    
    for model in ensemble_model.models:
        x_adv = PGD_attack(model, data, target, per_size, num_iter//2, step_size, type, bias=bias_data)
        x_advs.append(x_adv)
        # x_advs.append(data + bias_data)
    
    return data + bias_data, x_advs


def PGD_attack_ambi(ensemble_model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='linfty'):
    ensemble_model.eval()

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)
    eta.data = torch.zeros_like(ori_data).uniform_(-per_size / 2, per_size / 2)

    eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

    eta.requires_grad = True

    

    for i in range(num_iter):
        ambiguity = 0

        probability_adv = []
        for model in ensemble_model.models:
            output = model(data + eta)
            probability_adv.append(F.softmax(output, dim = 1))
        prob_average = 0
        
        for prob in probability_adv:
            prob_average = prob_average + prob
        prob_average = prob_average / len(probability_adv)

        for prob in probability_adv:
            ambiguity = ambiguity + F.kl_div(torch.log(prob), prob_average, reduction='batchmean')

        # breakpoint()

        ambiguity = ambiguity/len(ensemble_model.models)
        ambiguity.backward()

        if type == 'l2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view(batch_size, -1), p=2, dim=1)
            grad = grad/grad_norm.view(batch_size, 1, 1, 1)
            eta.data = eta.data + step_size*grad
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            grad_sign = eta.grad.data.sign()

            # abs_max = eta.grad.abs().max().data
            # grad = eta.grad/abs_max
            # grad_sign = 2*F.sigmoid(grad*500)-1

            eta.data = eta.data - step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        eta.grad.data.zero_()
    return ambiguity*target.size()[0]*len(ensemble_model.models)



def PGD_ensemble_attack(net_ls, inp, label, eps, nb_iter, sigma, weight=None, rand=True, inv=False):
        for net in net_ls:
            net.eval()
        if weight is None:
            weight = [1. for _ in net_ls]
        normalizing_constant = sum(weight) / len(weight)
        x = inp.detach()
        if rand:
            x = x + torch.zeros_like(inp).uniform_(-eps / 2, eps / 2)

        loss_func = torch.nn.CrossEntropyLoss()

        for _ in range(nb_iter):
            x.requires_grad_()
            with torch.enable_grad():
                pred = sum([net(x) * weight[i] for i, net in enumerate(net_ls)]) / len(net_ls)
                pred /= normalizing_constant
                loss = loss_func(pred, label)
                if inv:
                    loss = - loss

            grad_sign = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)[0].detach().sign()
            x = x.detach() + sigma * grad_sign
            x = torch.min(torch.max(x, inp - eps), inp + eps)
            # if self.natural:
            #     x = torch.clamp(x, 0., 1.)
        return x




def PGD_grad_norm(ensemble_model, data, target, per_size=8/255, num_iter=40, step_size = 0.005, type='l_infty', bias=None):

    ori_data = data.data
    batch_size = len(data)
    eta = torch.zeros_like(ori_data)
    eta = nn.Parameter(eta)

    rand_init_delta(eta, ori_data, type, per_size, clip_min = 0, clip_max = 1, bias = bias)

    eta.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(num_iter):
        
        grad_norm_1 = 0

        for base_model in ensemble_model.models:
            loss = loss_fn(base_model(data + eta), target)
            grad_1 = torch.autograd.grad(loss, eta, create_graph=True)[0]
            grad_1 = grad_1.flatten(start_dim=1)
            grad_1 = grad_1 * data.size()[0]
            grad_norm_1 += torch.norm(grad_1, p=2, dim=1)
        
        grad_norm_1 /= len(ensemble_model.models)
        grad_norm_1 = grad_norm_1.mean()

        grad_norm_1.backward()


        if type == 'l_2':
            grad = eta.grad
            grad_norm = torch.norm(grad.view([batch_size, -1]), p=2, dim=1)
            grad_norm = torch.max(grad_norm, (1e-6)*torch.ones_like(grad_norm))
            grad = grad/grad_norm.view([batch_size, 1, 1, 1])
            eta.data = eta.data + step_size*grad

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            eta_norm = torch.norm(eta.data.view([batch_size, -1]), p=2, dim=1)
            factor = torch.min(per_size / eta_norm, torch.ones_like(eta_norm))
            factor = factor.view([batch_size, 1, 1, 1])
            eta.data = factor * eta.data

            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data

            # eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        elif type == 'l_infty':
            grad_sign = eta.grad.data.sign()
            eta.data = eta.data + step_size*grad_sign
            eta.data = torch.clamp(eta.data, min=-per_size, max=per_size)
            eta.data = torch.clamp(ori_data+eta.data, min=0, max=1) - ori_data.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        eta.grad.data.zero_()
    
    return torch.clamp(ori_data+eta.data, min=0, max=1)
