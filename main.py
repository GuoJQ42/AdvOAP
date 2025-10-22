import torch
import torch.optim as optim
from show_diversity import test_and_show_the_diversity_two_class_regression, test_and_show_the_diversity_two_class_classification
from data.cifar10 import load_cifar10, load_cifar10_two_class_regression, load_cifar10_two_class_classification, load_MNIST, load_FashionMNIST
from data.cifar100 import load_cifar100
from models.resnet import ResNet18
from models.resnet20 import ResNet20

from trainer import Adv_Ensemble_Trainer
from tester import test_and_save_clean, test_and_save_adv

from torch.utils.data import DataLoader
from models.ensemble import EnsembleModel
from models.CNN import CNN_NET, CNN_NET_regression, CNN_NET_classification

import argparse
from tester import test_and_save_adv_two_class_regression, test_and_save_adv_two_class_classification, test_and_save_clean_two_class_regression, test_and_save_clean_two_class_classification



def train_OAP(device, alpha=0.2, beta=10):
    global TrainArgs
    global AttackArgs
    global EPOCH_NUM
    global LEARNING_RATE
    global ClASSIFIERS_NUM
    
    TrainArgs['alpha'] = alpha
    TrainArgs['beta'] = beta

    AttackArgs['method'] = 'adv_oap_%s_%d_%d_%.2f_%.2f'%(Dataset, EPOCH_NUM, ClASSIFIERS_NUM, TrainArgs['alpha'], TrainArgs['beta'])
    TrainArgs['method'] = 'adv_oap_%s_%d_%d_%.2f_%.2f'%(Dataset, EPOCH_NUM, ClASSIFIERS_NUM, TrainArgs['alpha'], TrainArgs['beta'])
    for idx in range(5):    
        train_model(idx, device)

def train_two_class_regression(device):
    global TrainArgs
    global AttackArgs
    global ClASSIFIERS_NUM
    global LEARNING_RATE
    global Dataset 
    global EPOCH_NUM 

    ClASSIFIERS_NUM = 1
    LEARNING_RATE = 0.01
    Dataset = 'CIFAR10_two_class_regression'
    EPOCH_NUM = 40
    
    AttackArgs['model'] = 'CNN_NET_regression'
    TrainArgs['model'] = 'CNN_NET_regression'
    AttackArgs['method'] = 'Two_class_train_regression_%d'%(EPOCH_NUM)
    TrainArgs['method'] = 'Two_class_train_regression_%d'%(EPOCH_NUM)
    
    for idx in range(20):
        train_model(idx, device)

def train_two_class_classification(device):
    global TrainArgs
    global AttackArgs
    global ClASSIFIERS_NUM
    global LEARNING_RATE
    global Dataset 
    global EPOCH_NUM 


    ClASSIFIERS_NUM = 1
    LEARNING_RATE = 1e-3
    Dataset = 'CIFAR10_two_class_classification'
    EPOCH_NUM = 60
    
    AttackArgs['model'] = 'CNN_NET_classification'
    TrainArgs['model'] = 'CNN_NET_classification'
    AttackArgs['method'] = 'Two_class_train_classification_%d'%(EPOCH_NUM)
    TrainArgs['method'] = 'Two_class_train_classification_%d'%(EPOCH_NUM)

    for idx in range(20):
        train_model(idx, device)



def train_model(idx, device):

    if Dataset == 'CIFAR10':
        train_dataset, test_dataset = load_cifar10('./data/data')
    elif Dataset == 'CIFAR10_two_class_regression':
        train_dataset, test_dataset = load_cifar10_two_class_regression(5, 6, './data/data')
    elif Dataset == 'CIFAR10_two_class_classification':
        train_dataset, test_dataset = load_cifar10_two_class_classification(5, 6, './data/data')
    elif Dataset == 'MNIST':
        train_dataset, test_dataset = load_MNIST('./data/data')
    elif Dataset == 'Fashion-MNIST':
        train_dataset, test_dataset = load_FashionMNIST('./data/data')
    elif Dataset == 'CIFAR100':
        train_dataset, test_dataset = load_cifar100('./data/data')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    if ClASSIFIERS_NUM == 1:
        if TrainArgs['model'] == 'ResNet20':
            ensemble_model = ResNet20().to(device)
        elif TrainArgs['model'] == 'ResNet18':
            if Dataset == 'CIFAR100':
                ensemble_model = ResNet18(100).to(device)
            else: 
                ensemble_model = ResNet18().to(device)
        elif TrainArgs['model'] == 'CNN':
            ensemble_model = CNN_NET().to(device)
        elif TrainArgs['model'] == 'CNN_NET_regression':
            ensemble_model = CNN_NET_regression().to(device)
        elif TrainArgs['model'] == 'CNN_NET_classification':
            ensemble_model = CNN_NET_classification().to(device)
    else:
        model_list = []
        for _ in range(ClASSIFIERS_NUM):
            if TrainArgs['model'] == 'ResNet20':
                model_list.append(ResNet20().to(device))
            elif TrainArgs['model'] == 'ResNet18':
                if Dataset == 'CIFAR100':
                    model_list.append(ResNet18(100).to(device))
                else: 
                    model_list.append(ResNet18().to(device))
            elif TrainArgs['model'] == 'CNN':
                model_list.append(CNN_NET().to(device))
            elif TrainArgs['model'] == 'CNN_NET_classification':
                model_list.append(CNN_NET_classification().to(device))
        ensemble_model = EnsembleModel(model_list)

    

    optimizer = optim.SGD(ensemble_model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    if Scheduler == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_NUM)
    else: 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones=[int(0.5 * EPOCH_NUM), int(0.75 * EPOCH_NUM)], gamma=0.1)

    print(TrainArgs['method'])

    best_acc = 0
    best_adv_acc = 0

    for epoch in range(EPOCH_NUM):
        
        Adv_Ensemble_Trainer(TrainArgs, ensemble_model, device, train_loader, optimizer, epoch, scheduler)

        if epoch % 20 == 0 or epoch >= EPOCH_NUM-10:
            if TrainArgs['model'] == 'CNN_NET_regression':
                best_acc = test_and_save_clean_two_class_regression(AttackArgs, ensemble_model, test_loader, device, epoch, best_acc, save = False)
                best_adv_acc, _ = test_and_save_adv_two_class_regression(AttackArgs, ensemble_model, test_loader, device, epoch, best_adv_acc, save = True, name = str(idx)+'_ckpt')
            elif TrainArgs['model'] == 'CNN_NET_classification':
                best_acc = test_and_save_clean_two_class_classification(ensemble_model, test_loader, device, epoch, best_acc, save = False)
                best_adv_acc, _ = test_and_save_adv_two_class_classification(AttackArgs, ensemble_model, test_loader, device, epoch, best_adv_acc, save = True, name = str(idx)+'_ckpt')
            else:
                best_acc = test_and_save_clean(AttackArgs, ensemble_model, test_loader, device, epoch, best_acc, save = False)
                best_adv_acc = test_and_save_adv(AttackArgs, ensemble_model, test_loader, device, epoch, best_adv_acc, save = True, name = str(idx)+'_ckpt')
    
    if TrainArgs['model'] == 'CNN_NET_regression':
        test_and_save_adv_two_class_regression(AttackArgs, ensemble_model, test_loader, device, 0, -1, save= True, name = str(idx)+'_last_ckpt')
    elif TrainArgs['model'] == 'CNN_NET_classification':
        test_and_save_adv_two_class_classification(AttackArgs, ensemble_model, test_loader, device, 0, -1, save = True, name = str(idx)+'_last_ckpt')
    else: 
        test_and_save_adv(AttackArgs, ensemble_model, test_loader, device, 0, -1, save= True, name = str(idx)+'_last_ckpt')


def test_model(device, path):

    if Dataset == 'MNIST':
        _, test_dataset = load_MNIST('./data/data')
    elif Dataset == 'Fashion-MNIST':
        _, test_dataset = load_FashionMNIST('./data/data')
    elif Dataset == 'CIFAR10':
        _, test_dataset = load_cifar10('./data/data')
    elif Dataset == 'CIFAR100':
        _, test_dataset = load_cifar100('./data/data')
    
    test_loader = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=1)
    
    adv_accs = []
    adv_acc_avg = []
    pred_vars = []

    for i in range(3):
        model_list = []
        adv_acc_subs = []
        
        for idx, model_idx in enumerate(random.sample([i for i in range(ClASSIFIERS_NUM)], ClASSIFIERS_NUM)):

            if Dataset == 'CIFAR100':
                model_list.append(ResNet18(100).to(device))
            else:
                model_list.append(ResNet20().to(device))

            print(model_idx)
            # saved = torch.load('./'+path+'/'+str(i)+'_ckpt_'+str(model_idx)+'.pth', weights_only=False)
            saved = torch.load('./'+path+'/'+str(i)+'_last_ckpt_'+str(model_idx)+'.pth', weights_only=False)
            # saved = torch.load('./'+path+'/'+str(i)+'_last_ckpt.pth', weights_only=False)
            model_dict = saved['net']
            model_list[idx].load_state_dict(model_dict)

            adv_acc_sub = 0
            adv_acc_subs.append(adv_acc_sub)

       
        ensemble_model = EnsembleModel(model_list)
            
        
        
      
        adv_acc = test_and_save_adv(AttackArgs, ensemble_model, test_loader, device, 0, 0, False, attack_method = AttackArgs['method'])

        adv_accs.append(adv_acc)
        adv_acc_avg.append(np.mean(adv_acc_subs))
    print('Adversarial Accuracy of Ensemble Model:')
    print(adv_accs)
    print('Adversarial Accuracy of Single Model:')
    print(adv_acc_avg)
    # print('Prediction Variance:')
    # print(pred_vars)
    print('Adv Acc Ensemble: %.4f+-%.4f, Adv Acc Single: %.4f+-%.4f'%(np.mean(adv_accs), np.std(adv_accs, ddof=1), np.mean(adv_acc_avg), np.std(adv_acc_avg, ddof=1)))
    return np.mean(adv_accs), np.std(adv_accs, ddof=1)

if __name__ == '__main__':

    # Default settings for CIFAR10
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--num_models', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--attack_type', type=str, default='l_infty')
    parser.add_argument('--perturbation_size', type=float, default=8/255)
    parser.add_argument('--perturb_steps_train', type=int, default=10)
    parser.add_argument('--perturb_steps_test', type=int, default=20)
    parser.add_argument('--step_size_train', type=float, default=1/255)
    parser.add_argument('--step_size_test', type=float, default=2/255)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--scheduler', type=str, default='Cosine')
    parser.add_argument('--model', type=str, default='ResNet20')
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--device_num', type=int, default=0)

    args = parser.parse_args()

    
    EPOCH_NUM = args.num_epochs
    ClASSIFIERS_NUM = args.num_models
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    Dataset = args.dataset
    AttackArgs = {'attack_type' :args.attack_type, 'perturbation_size' : args.perturbation_size, 'perturb_steps' : args.perturb_steps_train, 'step_size' : args.step_size_train}
    TrainArgs = {'attack_type' :args.attack_type, 'perturbation_size' : args.perturbation_size, 'perturb_steps' : args.perturb_steps_test, 'step_size' : args.step_size_test, 'num_classes': args.num_classes, 'batch_size': BATCH_SIZE, 'alpha': args.alpha, 'beta': args.beta}
    Scheduler = args.scheduler
    device = torch.device("cuda:%d"%(args.device_num) if torch.cuda.is_available() else "cpu")


    if args.mode == 'show_diversity':
        train_two_class_regression(device)
        train_two_class_classification(device)

        AttackArgs['model'] = 'CNN_NET_regression'
        TrainArgs['model'] = 'CNN_NET_regression'
        AttackArgs['method'] = 'Two_class_train_regression_40'
        TrainArgs['method'] = 'Two_class_train_regression_40'
        test_and_show_the_diversity_two_class_regression(AttackArgs, TrainArgs, device)

        AttackArgs['model'] = 'CNN_NET_classification'
        TrainArgs['model'] = 'CNN_NET_classification'
        AttackArgs['method'] = 'Two_class_train_classification_60'
        TrainArgs['method'] = 'Two_class_train_classification_60'
        test_and_show_the_diversity_two_class_classification(AttackArgs, TrainArgs, device)

    elif args.mode == 'train':
        AttackArgs['model'] = args.model
        TrainArgs['model'] = args.model
        train_OAP(device, args.alpha, args.beta)
    elif args.mode == 'test':
        path = 'checkpoint/ResNet20_adv_oap_250_8_0.20_10.00_models'
        AttackArgs['model'] = 'ResNet20'
        TrainArgs['model'] = 'ResNet20'
        print(path)
        AttackArgs['method'] = 'PGD'
        test_model(device, path)
    
    
