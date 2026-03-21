import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import random
from torch.optim import RMSprop
from typing import Optional
import scipy.io as scio
from torch.optim.optimizer import Optimizer
from Adversarial_DG import TripleDomainAdversarialLoss
from model_EEGMatch import Domain_adaption_model,discriminator_DG
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch SSL Training')

parser.add_argument('--dataset', default='SEED', type=str, metavar='N')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_video', default=3, type=int, metavar='N',
                    help='Number of label: SEED(3, 6, 9, 12), SEED-IV(4, 8, 12, 16, 20)')
parser.add_argument('--num_classes', default=3, type=int, metavar='N',
                    help='Number of label: SEED(3), SEED-IV(4)')
parser.add_argument('--momentum', default=0.999, type=float, metavar='N',
                    help='Momentum factor for updating confidence bank')
parser.add_argument('--tau', default=0.4, type=float, metavar='N',
                    help='Temperature scaling factor for pseudo-label adjustment')
parser.add_argument('--switch', default=0.5, type=float, metavar='N',
                    help='Switch batch iteration rate')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class StepwiseLR_GRL:
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75,max_iter: Optional[float] = args.epochs):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter
    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()    
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
#        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()

def augmentation(feature_seqence,label_seqence,video_time,alpha=0.5):
    augment_data=[]
    augment_label=[]
    flag=0
    if len(feature_seqence)==0:
        return feature_seqence,label_seqence
    for i in range(len(video_time)):
        video_feature=feature_seqence[flag:flag+video_time[i],:]
        video_label=label_seqence[flag:flag+video_time[i],:]
        for j in range(len(video_feature)):
            index=np.random.randint(0,len(video_feature),2)
            weight_sequence=0.5*np.ones(2).reshape((1,2))
            lam=np.random.beta(alpha,alpha)
            weight_sequence[0,0]=lam
            weight_sequence[0,1]=1-lam
            augment_data.append(np.dot(weight_sequence,video_feature[index,:]))
            augment_label.append(video_label[j,:])
        flag+=video_time[i]
    return np.vstack(augment_data),np.vstack(augment_label)

import numpy as np

def get_dataset_aug(test_id, session, video, parameter):
    alpha = parameter['alpha']

    feature_list_source_labeled = []
    label_list_source_labeled = []
    feature_list_source_unlabeled = []
    label_list_source_unlabeled = []
    feature_list_target = []
    label_list_target = []
    feature_list_source_labeled_aug = []
    label_list_source_labeled_aug = []
    feature_list_source_unlabeled_aug = []
    label_list_source_unlabeled_aug = []
    feature_list_target_aug = []
    label_list_target_aug = []

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

    path = 'your_data_path'
    video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]

    for subject_num in range(15):
        subject_id = subject_num + 1
        session_pattern = f"sub_{subject_id}_session_{session}.mat"
        file_path = os.path.join(path, session_pattern)

        mat_data = scio.loadmat(file_path)
        dataset_key = f'dataset_session{session}'
        feature = mat_data[dataset_key][0, 0]['feature']
        label = mat_data[dataset_key][0, 0]['label']

        if feature.ndim == 1:
            feature = feature.reshape(-1, 1)
        feature = min_max_scaler.fit_transform(feature).astype('float32')
        one_hot_label_mat = np.eye(args.num_classes)[label.flatten().astype(int)]

        trials = []
        start_idx = 0
        for length in video_time:
            end_idx = start_idx + length
            trials.append((feature[start_idx:end_idx, :], one_hot_label_mat[start_idx:end_idx, :]))
            start_idx = end_idx

        if subject_num != test_id:
            labeled_trials = trials[:video]  # First video trials for labeled data
            unlabeled_trials = trials[video:]  # Remaining trials for unlabeled data

            feature_labeled = np.concatenate([trial[0] for trial in labeled_trials], axis=0)
            label_labeled = np.concatenate([trial[1] for trial in labeled_trials], axis=0)
            feature_list_source_labeled.append(feature_labeled)
            label_list_source_labeled.append(label_labeled)

            feature_labeled_aug, label_labeled_aug = augmentation(feature_labeled, label_labeled, video_time[:video], alpha)
            feature_labeled = np.vstack((feature_labeled, feature_labeled_aug)).astype('float32')
            label_labeled = np.vstack((label_labeled, label_labeled_aug)).astype('float32')
            feature_list_source_labeled_aug.append(feature_labeled)
            label_list_source_labeled_aug.append(label_labeled)

            feature_unlabeled = np.concatenate([trial[0] for trial in unlabeled_trials], axis=0)
            label_unlabeled = np.concatenate([trial[1] for trial in unlabeled_trials], axis=0)
            feature_list_source_unlabeled.append(feature_unlabeled)
            label_list_source_unlabeled.append(label_unlabeled)

            feature_unlabeled_aug, label_unlabeled_aug = augmentation(feature_unlabeled, label_unlabeled, video_time[video:], alpha)
            feature_unlabeled = np.vstack((feature_unlabeled, feature_unlabeled_aug)).astype('float32')
            label_unlabeled = np.vstack((label_unlabeled, label_unlabeled_aug)).astype('float32')
            feature_list_source_unlabeled_aug.append(feature_unlabeled)
            label_list_source_unlabeled_aug.append(label_unlabeled)
        else:
            feature_list_target.append(feature)
            label_list_target.append(one_hot_label_mat)

            feature_aug, label_aug = augmentation(feature, one_hot_label_mat, video_time, alpha)
            feature = np.vstack((feature, feature_aug)).astype('float32')
            label = np.vstack((one_hot_label_mat, label_aug)).astype('float32')
            feature_list_target_aug.append(feature)
            label_list_target_aug.append(label)

    source_feature_labeled = np.vstack(feature_list_source_labeled)
    source_label_labeled = np.vstack(label_list_source_labeled)
    source_feature_unlabeled = np.vstack(feature_list_source_unlabeled)
    source_label_unlabeled = np.vstack(label_list_source_unlabeled)

    source_feature_labeled_aug = np.vstack(feature_list_source_labeled_aug)
    source_label_labeled_aug = np.vstack(label_list_source_labeled_aug)
    source_feature_unlabeled_aug = np.vstack(feature_list_source_unlabeled_aug)
    source_label_unlabeled_aug = np.vstack(label_list_source_unlabeled_aug)

    target_feature = feature_list_target[0]
    target_label = label_list_target[0]
    target_feature_aug = feature_list_target_aug[0]
    target_label_aug = label_list_target_aug[0]


    target_set = {'feature': target_feature, 'label': target_label, 'feature_aug': target_feature_aug,
                  'label_aug': target_label_aug}
    source_set_labeled = {'feature': source_feature_labeled, 'label': source_label_labeled,
                          'feature_aug': source_feature_labeled_aug, 'label_aug': source_label_labeled_aug}
    source_set_unlabeled = {'feature': source_feature_unlabeled, 'label': source_label_unlabeled,
                            'feature_aug': source_feature_unlabeled_aug, 'label_aug': source_label_unlabeled_aug}

    return target_set, source_set_labeled, source_set_unlabeled


def get_generated_targets(model,x_s,x_un,x_t,labels_s,semi):
        with torch.no_grad():
            model.eval()
            un_predict = model.predict_confidence(x_un, args.tau)
            t_predict = model.predict_confidence(x_t, args.tau)

            if semi==1:
                X,Y=torch.cat((x_s,x_un)),torch.cat((labels_s,un_predict.to(labels_s)))
            else:
                X,Y=x_s,labels_s

            _,_,_,_,_,dist_matrix_t,_,_ = model(X,x_t,Y.to(X),t_predict.to(x_t))
            sim_matrix = model.get_cos_similarity_distance(Y.to(X))
            sim_matrix_target = model.get_cos_similarity_by_threshold(dist_matrix_t)

            s_predict = model.predict(x_s)
            max_values = torch.max(s_predict, dim=1).values
            qhat_threshold = torch.mean(max_values)
            qhat_mask_source_labeled = (max_values > qhat_threshold).float()
            model.update_qhat(s_predict, model.qhat, momentum=args.momentum, qhat_mask=qhat_mask_source_labeled)

            max_values_un = torch.max(un_predict, dim=1).values
            qhat_threshold_un = torch.mean(max_values_un)
            qhat_mask_source = (max_values_un > qhat_threshold_un).float()
            model.update_qhat(un_predict, model.qhat, momentum=args.momentum, qhat_mask=qhat_mask_source)

            max_values_target = torch.max(t_predict, dim=1).values
            qhat_threshold_target = torch.mean(max_values_target)
            qhat_mask_target = (max_values_target > qhat_threshold_target).float()
            model.update_qhat(t_predict, model.qhat, momentum=args.momentum, qhat_mask=qhat_mask_target)

            return sim_matrix,sim_matrix_target,un_predict,t_predict

def checkpoint(model,checkpoint_PATH,flag):
    if flag=='load':
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        model.P=model_CKPT['P']
        model.stored_mat=model_CKPT['stored_mat']
        model.cluster_label=model_CKPT['cluster_label']
        model.upper_threshold=model_CKPT['upper_threshold']
        model.lower_threshold=model_CKPT['lower_threshold']
        model.threshold=model_CKPT['threshold']
    elif flag=='save':
        torch.save({'P': model.P, 'stored_mat':model.stored_mat,'cluster_label':model.cluster_label,'threshold':model.threshold,
                    'upper_threshold':model.upper_threshold,'lower_threshold':model.lower_threshold,'state_dict': model.state_dict()},checkpoint_PATH)


def train_model(loader_train_labeled,loader_train_unlabeled, loader_test,model,dann_loss, optimizer,hidden_4,epoch,batch_size,parameter,threshold_update=True):
    model.train()
    dann_loss.train()
    train_source_iter_labeled,train_source_iter_unlabeled,train_target_iter=enumerate(loader_train_labeled),enumerate(loader_train_unlabeled),enumerate(loader_test)
    T =2*3394//batch_size
    cls_loss_sum=0
    transfer_loss_sum=0
    cluster_loss_sum=0
    if parameter['boost_type']=='linear':
        boost_factor=parameter['cluster_weight']*(epoch/model.max_iter)
    elif parameter['boost_type']=='exp':
        boost_factor=parameter['cluster_weight']*(2.0 / (1.0 + np.exp(-1 * epoch / model.max_iter))- 1)
    elif parameter['boost_type']=='constant':
        boost_factor=parameter['cluster_weight']
    switch_phase_called = False
    switch_epoch = int(args.switch * T)
    if args.tau < 0:
        args.tau *= (-1)
    for i in range(T):
        model.train()
        _,(x_s,labels_s) = next(train_source_iter_labeled)
        _,(x_un,_) = next(train_source_iter_unlabeled)
        _,(x_t,_) = next(train_target_iter)
        x_t=Variable(x_t.cuda())
        x_s,labels_s,x_un=Variable(x_s.cuda()), Variable(labels_s.cuda()),Variable(x_un.cuda())
        with torch.no_grad():
            estimated_sim_truth,estimated_sim_truth_target,un_predict,t_predict= get_generated_targets(model,x_s,x_un,x_t,labels_s,1)
            X,Y=torch.cat((x_s,x_un)),torch.cat((labels_s,un_predict.to(labels_s)))
        source_predict,target_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target,source_label_feature,target_label_feature = model(X,x_t,Y.to(X),t_predict.to(x_t))
        eta=0.00001
        sim_matrix = torch.clamp(sim_matrix, min=eta, max=1.0 - eta)
        sim_matrix_target = torch.clamp(sim_matrix_target, min=eta, max=1.0 - eta)
        bce_loss=-(torch.log(sim_matrix+eta)*estimated_sim_truth)-(1-estimated_sim_truth)*torch.log(1-sim_matrix+eta)
        bce_loss_target=-(torch.log(sim_matrix_target+eta)*estimated_sim_truth_target)-(1-estimated_sim_truth_target)*torch.log(1-sim_matrix_target+eta)
        indicator,nb_selected=model.compute_indicator(sim_matrix_target)
        cls_loss = torch.mean(bce_loss)
        nb_selected = max(1, torch.sum(indicator).item())
        cluster_loss=torch.sum(indicator*bce_loss_target)/nb_selected
        P_loss=torch.norm(torch.matmul(model.P.T,model.P)-torch.eye(hidden_4).cuda(),'fro')
        transfer_loss = dann_loss(feature_source_f[0:len(x_s),:],feature_target_f,feature_source_f[len(x_s):len(feature_source_f),:])
        cls_loss_sum+=cls_loss.data
        transfer_loss_sum+=transfer_loss.data
        cluster_loss_sum+=cluster_loss.data
        loss = cls_loss+transfer_loss+0.01*P_loss+boost_factor*cluster_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i >= switch_epoch and not switch_phase_called:
            args.tau *= (-1)
    if threshold_update==True:
        model.update_threshold(epoch)  
    return cls_loss_sum.cpu().detach().numpy(),transfer_loss_sum.cpu().detach().numpy(),cluster_loss_sum.cpu().detach().numpy(), cls_loss, transfer_loss, P_loss, cluster_loss, loss

def train_and_test_GAN(test_id,max_iter,parameter,session,threshold_update=True):
    setup_seed(20)
    hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,upper_threshold,lower_threshold,temp=parameter['hidden_1'],parameter['hidden_2'], parameter['hidden_1'],parameter['hidden_2'],parameter['num_of_class'],parameter['low_rank'],parameter['upper_threshold'],parameter['lower_threshold'],parameter['temp']
    BATCH_SIZE = parameter['batch_size']
    video=parameter['video']
    target_set,source_set_labeled,source_set_unlabeled=get_dataset_aug(test_id,session,video,parameter)
    torch_dataset_source_labeled = Data.TensorDataset(torch.from_numpy(source_set_labeled['feature_aug']),torch.from_numpy(source_set_labeled['label_aug']))
    torch_dataset_source_unlabeled = Data.TensorDataset(torch.from_numpy(source_set_unlabeled['feature_aug']),torch.from_numpy(source_set_unlabeled['label_aug']))
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature_aug']),torch.from_numpy(target_set['label_aug']))
    test_features,test_labels=torch.from_numpy(target_set['feature']),torch.from_numpy(target_set['label'])
    source_features,source_labels=torch.from_numpy(source_set_labeled['feature']),torch.from_numpy(source_set_labeled['label'])
    valid_features,valid_labels=torch.from_numpy(source_set_unlabeled['feature']),torch.from_numpy(source_set_unlabeled['label'])
    BATCH_SIZE_UN=BATCH_SIZE
    loader_train_labeled = Data.DataLoader(
            dataset=torch_dataset_source_labeled,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
            )
    loader_train_unlabeled = Data.DataLoader(
        dataset=torch_dataset_source_unlabeled,
        batch_size=BATCH_SIZE_UN,
        shuffle=True,
        num_workers=0
        )
    BATCH_SIZE_Target=BATCH_SIZE
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=BATCH_SIZE_Target,
            shuffle=True,
            num_workers=0
            ) 
    setup_seed(20)
    model=Domain_adaption_model(hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold, temp).cuda(0)
    model.apply(weigth_init)
    domain_discriminator = discriminator_DG(hidden_2).cuda()
    domain_discriminator.apply(weigth_init)
    dann_loss = TripleDomainAdversarialLoss(domain_discriminator).cuda()
    optimizer = RMSprop(model.get_parameters() + domain_discriminator.get_parameters(),lr=1e-3, weight_decay=1e-5)
    best_acc = 0.
    target_acc_list=np.zeros(max_iter)
    target_nmi_list=np.zeros(max_iter)
    source_acc_list=np.zeros(max_iter)
    source_nmi_list=np.zeros(max_iter)
    cls_loss_list=np.zeros(max_iter)
    transfer_loss_list=np.zeros(max_iter)
    pbar_train = tqdm(range(max_iter),
                      total=len(range(max_iter)),
                      desc=f'[Training]',
                      ncols=100,
                      ascii=' #',
                      leave=False,
                      )
    for epoch in pbar_train:
        if (len(np.unique(model.cluster_label))!=3):
            model.cluster_label=np.hstack([0,1,2])
        model.train()
        cls_loss_sum,transfer_loss_sum, cluster_loss_sum, cls_loss, transfer_loss, P_loss, cluster_loss, loss=train_model(loader_train_labeled,loader_train_unlabeled,loader_test,model,dann_loss,optimizer,hidden_4,epoch,BATCH_SIZE_Target,parameter,threshold_update)
        source_acc,source_nmi=model.cluster_label_update(source_features.cuda(),source_labels.cuda())
        model.eval()
        target_acc,target_nmi=model.target_domain_evaluation(test_features.cuda(),test_labels.cuda())
        valid_acc,valid_nmi=model.target_domain_evaluation(valid_features.cuda(),valid_labels.cuda())
        target_acc_list[epoch]=target_acc
        source_acc_list[epoch]=source_acc
        target_nmi_list[epoch]=target_nmi
        source_nmi_list[epoch]=source_nmi
        cls_loss_list[epoch]=cls_loss_sum
        transfer_loss_list[epoch]=transfer_loss_sum

        pbar_train.set_description(
            f"[{epoch}/{max_iter}] SRC ACC: {round(source_acc, 4)} TAR ACC: {round(target_acc, 4)}, VAL ACC: {round(valid_acc, 4)}"
        )
        if target_acc > best_acc:
            best_acc = target_acc
    return best_acc,cls_loss_list,source_acc_list,source_nmi_list,target_acc_list,target_nmi_list,transfer_loss_list, model

def main(update_threshold,parameter,session):
    setup_seed(20)
    max_iter=args.epochs
    best_acc_mat=np.zeros(15)
    transfer_loss_curve=np.zeros((15,max_iter))
    cls_loss_curve=np.zeros((15,max_iter))
    source_acc_curve=np.zeros((15,max_iter))
    target_acc_curve=np.zeros((15,max_iter))
    source_nmi_curve=np.zeros((15,max_iter))
    target_nmi_curve=np.zeros((15,max_iter))
    for i in range(15):
        best_acc,cls_loss_list,source_acc_list,source_nmi_list,target_acc_list,target_nmi_list,transfer_loss_list, model=train_and_test_GAN(i,max_iter,parameter,session,update_threshold)
        best_acc_mat[i]=best_acc
        source_acc_curve[i,:]=source_acc_list
        target_acc_curve[i,:]=target_acc_list
        source_nmi_curve[i,:]=source_nmi_list
        target_nmi_curve[i,:]=target_nmi_list
        transfer_loss_curve[i,:]=transfer_loss_list
        cls_loss_curve[i,:]=cls_loss_list

    return best_acc_mat,cls_loss_curve,transfer_loss_curve,source_acc_curve,source_nmi_curve,target_acc_curve,target_nmi_curve

directory = './my_result'

video = args.num_video
parameter={'hidden_1':64,'hidden_2':64,'num_of_class':3,'cluster_weight':2,'low_rank':32,'upper_threshold':0.9,'lower_threshold':0.5,
           'boost_type':'linear','video':video,'temp':0.9,'batch_size':48,'alpha':0.5}

best_acc_mat,cls_loss_curve,transfer_loss_curve,source_acc_curve,source_nmi_curve,target_acc_curve,target_nmi_curve=main(True,parameter,1)
result_list={'best_acc_mat':best_acc_mat,
            'cls_loss_curve':cls_loss_curve,
            'source_acc_curve':source_acc_curve,
            'source_nmi_curve':source_nmi_curve,
            'target_acc_curve':target_acc_curve,
            'target_nmi_curve':target_nmi_curve}

np.savetxt(os.path.join(directory, f'best_acc_mat_labeled.csv'),
           best_acc_mat, delimiter=",", fmt="%.4f")
np.savetxt(os.path.join(directory, f'cls_loss_curve_labeled.csv'),
           cls_loss_curve, delimiter=",", fmt="%.4f")
np.savetxt(os.path.join(directory, f'transfer_loss_curve_labeled.csv'),
           transfer_loss_curve, delimiter=",", fmt="%.4f")
np.savetxt(os.path.join(directory, f'source_acc_curve_labeled.csv'),
           source_acc_curve, delimiter=",", fmt="%.4f")
np.savetxt(os.path.join(directory, f'source_nmi_curve_labeled.csv'),
           source_nmi_curve, delimiter=",", fmt="%.4f")
np.savetxt(os.path.join(directory, f'target_acc_curve_labeled.csv'),
           target_acc_curve, delimiter=",", fmt="%.4f")
np.savetxt(os.path.join(directory, f'target_nmi_curve_labeled.csv'),
           target_nmi_curve, delimiter=",", fmt="%.4f")
