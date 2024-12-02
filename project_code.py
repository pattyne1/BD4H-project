""" ########## Setup ########## """

# Libraries
import os

# Gets directory where script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, "data_synthetic"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)

# Change to the base directory
os.chdir(BASE_DIR)

# Print working directory to verify
print("Working Directory:", os.getcwd())

# Define paths used throughout the code
DATA_DIR = os.path.join(BASE_DIR, "data_synthetic")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


""" ########## Simulated Data ########## """

# Libraries 
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# timestamp
T = 30
# num of covariates
k = 100
# num of static features
k_s = 5
# num of hidden
h = 1
# num of samples
N = 4000
N_treated = 1000
# seed
np.random.seed(66)
# num of P
p = 5
# weight of hidden confounders
gamma_h = 0.1
# bias
kappa = 10

S = 20

# Define directories for saving data
dir = os.path.join(DATA_DIR, f'data_syn_{gamma_h}')
dir_base = os.path.join(DATA_DIR, f'data_baseline_syn_{gamma_h}')
os.makedirs(dir, exist_ok=True)
os.makedirs(dir_base, exist_ok=True)


eta, epsilon = np.random.normal(0,0.001, size=(N,T,k)),np.random.normal(0,0.001, size=(N,T,h))
w = np.random.uniform(-1, 1, size=(h+1, 2))
b = np.random.normal(0, 0.1, size=(N, 2))


# 1. simulate treatment A
A = np.zeros(shape=(N, T))
for n in range(N_treated):
    initial_point = np.random.choice(range(T))
    a = np.zeros(T)
    a[initial_point:] = 1
    A[n] = a

np.random.shuffle(A)


# 2. simulate covariates X and hidden confounders Z using A
# 3. simulate outcome Y
X = np.random.normal(0, 0.5, size=(N,k))
X[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.5, size=(N_treated, k))
X_static = np.random.normal(0, 0.5, size=(N,k_s))
X_static[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.5, size=(N_treated, k_s))
Z = np.random.normal(0, 0.5, size=(N,h))
Z[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.5, size=(N_treated, h))
delta = np.random.uniform(-1, 1, size=(k + k_s, h))

A_final = np.where(np.sum(A, axis=1)>0, 1, 0)

X_all, Z_all = [X], [Z]
for t in range(1, T+1):
    i = 1
    tmp_x = 0
    tmp_z = 0
    while (t-i) >= 0 and i <= p:
        alpha = np.random.normal(1 - (i / p), (1 / p), size=(N, k))
        beta = np.random.normal(0, 0.02, size=(N, k))
        beta[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.02, size=(N_treated, k))
        tmp_x += np.multiply(alpha, X_all[t - i]) + np.multiply(beta, np.tile(A[:, t - i], (k, 1)).T)

        mu = np.random.normal(1 - (i / p), (1 / p), size=(N, h))
        v = np.random.normal(0, 0.02, size=(N, h))
        v[np.where(np.sum(A, axis=1) > 0), :] = np.random.normal(1, 0.02, size=(N_treated, h))
        tmp_z += np.multiply(mu, Z_all[t - i]) + np.multiply(v, np.tile(A[:, t - i], (h, 1)).T)
        i += 1

    X = tmp_x/(i-1) + eta[:,t-1,:]
    Z = tmp_z/(i-1) + epsilon[:,t-1,:]

    X_all.append(X)
    Z_all.append(Z)

    Q = gamma_h * Z + (1-gamma_h) * np.expand_dims(np.mean(np.concatenate((X, X_static), axis=1), axis=1), axis=1)

w = np.random.uniform(-1, 1, size=(1, 2))
b = np.random.normal(0, 0.1, size=(N, 2))
Y = np.matmul(Q, w) + b
Y_f = A_final * Y[:,0] + (1-A_final) * Y[:,1]
Y_cf = A_final * Y[:,1] + (1-A_final) * Y[:,0]


for n in tqdm(range(N)):
    x = np.zeros(shape=(T, k))
    out_x_file = os.path.join(dir, f'{n}.x.npy')
    out_static_file = os.path.join(dir, f'{n}.static.npy')
    out_a_file = os.path.join(dir, f'{n}.a.npy')
    out_y_file = os.path.join(dir, f'{n}.y.npy')
    
    for t in range(1, T+1):
        x[t-1, :] = X_all[t][n,:]
    x_static = X_static[n,:]
    a = A[n,:]

    y = [Y_f[n], Y_cf[n]]

    try: 
        np.save(out_x_file, x)
        np.save(out_static_file, x_static)
        np.save(out_a_file, a)
        np.save(out_y_file, y)
    except Exception as e:
        print(f"Error saving files for sample {n}: {e}")

all_idx = np.arange(N)
np.random.shuffle(all_idx)

train_ratio = 0.7
val_ratio = 0.1

train_idx = all_idx[:int(len(all_idx)*train_ratio)]
val_idx = all_idx[int(len(all_idx) * train_ratio):int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio)]
test_idx = all_idx[int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio):]

split = np.ones(N)
split[test_idx] = 0
split[val_idx] = 2

df = pd.DataFrame(split, dtype=int)
df.to_csv(os.path.join(dir, 'train_test_split.csv'), index=False, header=False)

for t in tqdm(range(1, T+1)):
    # a + y_f + y_cf + n_covariates + split

    out_matrix = np.zeros((N, k+k_s+1+2+1))

    out_matrix[:,0] = A_final
    out_matrix[:,3:3+k] = X_all[t]
    out_matrix[:,3+k:3+k+k_s] = X_static

    out_matrix[:,1] = Y_f
    out_matrix[:,2] = Y_cf

    out_matrix[:,-1] = split

    df = pd.DataFrame(out_matrix)
    df.to_csv(os.path.join(dir_base, f'{t}.csv'), index=False)



"""########## Load Data ##########"""

# Libraries
import numpy as np
import torch
import os
from torch.utils import data

gamma=0.1

data_dir = os.path.join(DATA_DIR, f'data_syn_{gamma}')


# dataset meta data
n_X_features = 100
n_X_static_features = 5
n_X_t_types = 1
n_classes = 1


def get_dim():
    return n_X_features, n_X_static_features, n_X_t_types, n_classes


class SyntheticDataset(data.Dataset):
    def __init__(self, list_IDs, obs_w, treatment):
        '''Initialization'''
        self.list_IDs = list_IDs
        self.obs_w = obs_w
        self.treatment = treatment


    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''

        try: 
            # Select sample
            ID = self.list_IDs[index]

            # Load labels
            label = np.load(os.path.join(data_dir, f'{ID}.y.npy'))

            # Load data
            X_demographic = np.load(os.path.join(data_dir, f'{ID}.static.npy'))
            X_all = np.load(os.path.join(data_dir, f'{ID}.x.npy'))
            X_treatment_res = np.load(os.path.join(data_dir, f'{ID}.a.npy'))
        
            # Convert to torch tensors
            X = torch.from_numpy(X_all.astype(np.float32))
            X_demo = torch.from_numpy(X_demographic.astype(np.float32))
            X_treatment = torch.from_numpy(X_treatment_res.astype(np.float32))
            y = torch.from_numpy(label.astype(np.float32))
            
            return X, X_demo, X_treatment, y

            
        except Exception as e:
            print(f"Error loading data for ID {ID}: {e}")
            raise


"""########### Define Model ##########"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat','concat2']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'concat2':
            self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def concat_score2(self, hidden, encoder_output):
        h = torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
        h = torch.cat((h, hidden*encoder_output),2)
        energy = self.attn(h).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'concat2':
            attn_energies = self.concat_score2(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LSTMModel(nn.Module):
    def __init__(self, n_X_features, n_X_static_features, n_X_fr_types, n_Z_confounders,
                 attn_model, n_classes, obs_w,
                 batch_size, hidden_size,
                 num_layers=2, bidirectional=True, dropout = 0.2):
        super().__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_X_features = n_X_features
        self.n_X_static_features = n_X_static_features
        self.n_classes = n_classes
        self.obs_w = obs_w
        self.num_layers = num_layers
        self.x_emb_size = 32
        self.x_static_emb_size = 16
        self.z_dim = n_Z_confounders

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.n_t_classes = 1

        self.rnn_f = nn.GRUCell(input_size=self.x_emb_size + 1 + n_Z_confounders, hidden_size=hidden_size)
        self.rnn_cf = nn.GRUCell(input_size=self.x_emb_size + 1 + n_Z_confounders, hidden_size=hidden_size)

        self.attn_f = Attn(attn_model, hidden_size)
        self.concat_f = nn.Linear(hidden_size * 2, hidden_size)

        self.attn_cf = Attn(attn_model, hidden_size)
        self.concat_cf = nn.Linear(hidden_size * 2, hidden_size)



        self.x2emb = nn.Linear(n_X_features, self.x_emb_size)
        self.x_static2emb = nn.Linear(n_X_static_features, self.x_static_emb_size)

        # IPW
        self.hidden2hidden_ipw = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.x_emb_size + hidden_size + self.x_static_emb_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.hidden2out_ipw = nn.Linear(hidden_size, self.n_t_classes, bias=False)

        # Outcome
        self.hidden2hidden_outcome_f = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear((self.x_emb_size + hidden_size) + self.x_static_emb_size + 1, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.hidden2out_outcome_f = nn.Linear(hidden_size, self.n_classes, bias=False)

        self.hidden2hidden_outcome_cf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.x_emb_size + hidden_size + self.x_static_emb_size + 1, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.hidden2out_outcome_cf = nn.Linear(hidden_size, self.n_classes, bias=False)


    def feature_encode(self, x, x_fr):

        f_hx = torch.randn(x.size(0), self.hidden_size)
        cf_hx = torch.randn(x.size(0), self.hidden_size)
        f_old = f_hx
        cf_old = cf_hx
        f_outputs = []
        f_zxs = []
        cf_outputs = []
        cf_zxs = []
        for i in range(x.size(1)):
            x_emb = self.x2emb(x[:, i, :])
            f_zx = torch.cat((x_emb, f_old), -1)
            f_zxs.append(f_zx)

            cf_zx = torch.cat((x_emb, cf_old), -1)
            cf_zxs.append(cf_zx)

            f_inputs = torch.cat((f_zx, x_fr[:,i].unsqueeze(1)), -1)

            cf_treatment = torch.where(x_fr.sum(1)==0, torch.Tensor([1]), torch.Tensor([0])).unsqueeze(1)
            cf_inputs = torch.cat((cf_zx, cf_treatment), -1)

            f_hx = self.rnn_f(f_inputs, f_hx)
            cf_hx = self.rnn_cf(cf_inputs, cf_hx)

            if i == 0:
                f_concat_input = torch.cat((f_hx, f_hx), 1)
                cf_concat_input = torch.cat((cf_hx, cf_hx), 1)
            else:
                f_attn_weights = self.attn_f(f_hx, torch.stack(f_outputs))
                f_context = f_attn_weights.bmm(torch.stack(f_outputs).transpose(0, 1))
                f_context = f_context.squeeze(1)
                f_concat_input = torch.cat((f_hx, f_context), 1)

                cf_attn_weights = self.attn_cf(cf_hx, torch.stack(cf_outputs))
                cf_context = cf_attn_weights.bmm(torch.stack(cf_outputs).transpose(0, 1))
                cf_context = cf_context.squeeze(1)
                cf_concat_input = torch.cat((cf_hx, cf_context), 1)

            f_concat_output = torch.tanh(self.concat_f(f_concat_input))
            f_old = f_concat_output

            cf_concat_output = torch.tanh(self.concat_cf(cf_concat_input))
            cf_old = cf_concat_output

            f_outputs.append(f_hx)
            cf_outputs.append(cf_hx)

        return f_zxs, cf_zxs


    def forward(self, x, x_demo, x_fr):

        f_zxs, cf_zxs = self.feature_encode(x, x_fr)

        # IPW
        ipw_outputs = []
        x_demo_emd = self.x_static2emb(x_demo)
        for i in range(len(f_zxs)):
            h = torch.cat((f_zxs[i], x_demo_emd), -1)
            h = self.hidden2hidden_ipw(h)
            ipw_out = self.hidden2out_ipw(h)
            ipw_outputs.append(ipw_out)


        # Outcome
        f_treatment = torch.where(x_fr.sum(1) > 0, torch.Tensor([1]), torch.Tensor([0])).unsqueeze(1)
        cf_treatment = torch.where(x_fr.sum(1) > 0, torch.Tensor([0]), torch.Tensor([1])).unsqueeze(1)

        # factual prediction

        f_zx_maxpool = torch.max(torch.stack(f_zxs), 0)

        f_hidden = torch.cat((f_zx_maxpool[0], x_demo_emd, f_treatment), -1)
        f_h = self.hidden2hidden_outcome_f(f_hidden)

        f_outcome_out = self.hidden2out_outcome_f(f_h)

        # counterfactual prediction

        cf_zx_maxpool = torch.max(torch.stack(cf_zxs), 0)

        cf_hidden = torch.cat((cf_zx_maxpool[0], x_demo_emd, cf_treatment), -1)
        cf_h = self.hidden2hidden_outcome_cf(cf_hidden)

        cf_outcome_out = self.hidden2out_outcome_cf(cf_h)


        return ipw_outputs, f_outcome_out, cf_outcome_out, f_h

"""########## Train on Synthetic Data ########## """

# Libraries 
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error

HIDDEN_SIZE = 32
CUDA = False


def trainInitIPTW(train_loader, val_loader,test_loader, model, epochs, optimizer, criterion,
                  l1_reg_coef=None, use_cuda=False, save_model=None):

    if use_cuda:
        print("====> Using CUDA device: ", torch.cuda.current_device(), flush=True)
        model.cuda()
        model = model.to('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Train network
    best_pehe_val = float('inf')
    best_loss_val = float('inf')
    best_pehe_test = float('inf')
    best_ate_test = float('inf')
    best_mse_test = float('inf')
    for epoch in range(epochs):
        ipw_epoch_losses = []
        outcome_epoch_losses = []
        f_train_outcomes = []
        f_train_treatments = []

        for x_inputs, x_static_inputs, x_fr_inputs, targets in tqdm(train_loader):
            model.train()

            # train IPW
            optimizer.zero_grad()

            fr_targets = x_fr_inputs
            if use_cuda:
                x_inputs, x_static_inputs, x_fr_inputs = x_inputs.cuda(), x_static_inputs.cuda(), x_fr_inputs.cuda()
                targets, fr_targets = targets.cuda(), fr_targets.cuda()

            ipw_outputs, f_outcome_out, cf_outcome_out, _ = model(x_inputs, x_static_inputs, fr_targets)
            f_treatment = torch.where(fr_targets.sum(1) > 0, torch.Tensor([1]), torch.Tensor([0]))

            f_train_outcomes.append(targets[:,0])
            f_train_treatments.append(f_treatment)

            ipw_loss = 0

            for i in range(len(ipw_outputs)):
                ipw_pred_norm = ipw_outputs[i].squeeze(1)
                ipw_loss += criterion(ipw_pred_norm, fr_targets[:, i].float())

            ipw_loss = ipw_loss/len(ipw_outputs)

            weights = torch.zeros(len(ipw_outputs[-1]))
            treat_sum = torch.sum(fr_targets, axis=1)
            p_treated = torch.where(treat_sum == 0)[0].size(0) / treat_sum.size(0)

            ipw_outputs = torch.cat(ipw_outputs, dim=1)
            ps = torch.sigmoid(ipw_outputs)

            for i in range(len(ps)):
                for t in range(ipw_outputs.size(1)):
                    if treat_sum[i] != 0:
                        weights[i] += p_treated / ps[i, t]
                    else:
                        weights[i] += (1 - p_treated) / (1 - ps[i, t])

            weights = weights / ipw_outputs.size(1)

            weights = torch.where(weights >= 100, torch.Tensor([100]), weights)
            weights = torch.where(weights <= 0.01, torch.Tensor([0.01]), weights)

            outcome_loss = torch.mean(weights*(f_outcome_out - targets[:,0]) ** 2)

            loss = ipw_loss * 0.05 + outcome_loss

            if l1_reg_coef:
                l1_regularization = torch.zeros(1)
                for pname, param in model.hidden2hidden_ipw.named_parameters():
                    if 'weight' in pname:
                        l1_regularization += torch.norm(param, 1)
                for pname, param in model.hidden2out_outcome_f.named_parameters():
                    if 'weight' in pname:
                        l1_regularization += torch.norm(param, 1)
                loss += (l1_reg_coef * l1_regularization).squeeze()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            optimizer.step()
            ipw_epoch_losses.append(ipw_loss.item())
            outcome_epoch_losses.append(loss.item())


        epoch_losses_ipw = np.mean(ipw_epoch_losses)
        outcome_epoch_losses = np.mean(outcome_epoch_losses)


        print('Epoch: {}, IPW train loss: {}'.format(epoch, epoch_losses_ipw), flush=True)
        print('Epoch: {}, Outcome train loss: {}'.format(epoch, outcome_epoch_losses), flush=True)


        # validation
        print('Validation:')

        pehe_val, _, mse_val, loss_val = model_eval(model, val_loader, criterion, eval_use_cuda=use_cuda)

        # if pehe_val < best_pehe_val:
        #     best_pehe_val = pehe_val

        if loss_val < best_loss_val:
            best_loss_val = loss_val

            if save_model:
                print('Best model. Saving...\n')
                torch.save(model, save_model)

                print('Test:')
                pehe_test,ate_test,mse_test,_ = model_eval(model, test_loader,criterion, eval_use_cuda=use_cuda)
                best_pehe_test = pehe_test
                best_ate_test = ate_test
                best_mse_test = mse_test

    print(np.sqrt(best_pehe_test))
    print(best_ate_test)
    print(np.sqrt(best_mse_test))
    return best_pehe_test


def transfer_data(model, dataloader, criterion, eval_use_cuda=False):
    with torch.no_grad():
        model.eval()
        f_outcome_outputs = []
        cf_outcome_outputs = []
        f_outcome_true = []
        cf_outcome_true = []
        ipw_true = []
        loss_all = []

        for x_inputs, x_static_inputs, x_fr_inputs, targets in dataloader:
            fr_targets = x_fr_inputs
            if eval_use_cuda:
                x_inputs, x_static_inputs, x_fr_inputs = x_inputs.cuda(), x_static_inputs.cuda(), x_fr_inputs.cuda()
                targets, fr_targets = targets.cuda(), fr_targets.cuda()


            ipw_outputs, f_outcome_out, cf_outcome_out, _ = model(x_inputs, x_static_inputs, fr_targets)

            ipw_loss = 0
            for i in range(len(ipw_outputs)):
                ipw_pred_norm = ipw_outputs[i].squeeze(1)
                ipw_loss += criterion(ipw_pred_norm, fr_targets[:, i].float())


            outcome_loss = torch.mean((f_outcome_out - targets[:,0]) ** 2)

            loss = ipw_loss * 0.05 + outcome_loss


            if eval_use_cuda:
                for i in range(len(ipw_outputs)):
                    ipw_outputs[i]=ipw_outputs[i].to('cpu').detach().data.numpy()
                fr_targets = fr_targets.to('cpu').detach().data.numpy()
                targets = targets.to('cpu').detach().data.numpy()
                f_outcome_out = f_outcome_out.to('cpu').detach().data.numpy()
                cf_outcome_out = cf_outcome_out.to('cpu').detach().data.numpy()
                loss = loss.to('cpu').detach().data.numpy()
            else:
                for i in range(len(ipw_outputs)):
                    ipw_outputs[i]=ipw_outputs[i].detach().data.numpy()
                ipw_outputs = ipw_outputs.detach().data.numpy() 
                x_fr_inputs = x_fr_inputs.detach().data.numpy()
                targets = targets.detach().data.numpy()
                outcome_outputs = outcome_outputs.detach().data.numpy()

            ipw_true.append(np.where(fr_targets.sum(1) > 0, 1, 0))
            f_outcome_true.append(targets[:,0])
            cf_outcome_true.append(targets[:, 1])
            f_outcome_outputs.append(f_outcome_out)
            cf_outcome_outputs.append(cf_outcome_out)
            loss_all.append(loss)


        ipw_true = np.concatenate(ipw_true).transpose()
        f_outcome_true = np.concatenate(f_outcome_true)
        cf_outcome_true = np.concatenate(cf_outcome_true)
        f_outcome_outputs = np.concatenate(f_outcome_outputs)
        cf_outcome_outputs = np.concatenate(cf_outcome_outputs)
        # loss_all = np.concatenate(loss_all)
        loss_all = np.mean(loss_all)

        return ipw_true, f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs, loss_all


def compute_pehe_ate(t, y_f, y_cf, y_pred_f, y_pred_cf):

    y_treated_true = t * y_f + (1-t) * y_cf
    y_control_true = t * y_cf + (1 - t) * y_f

    y_treated_pred = t * y_pred_f + (1 - t) * y_pred_cf
    y_control_pred = t * y_pred_cf + (1 - t) * y_pred_f

    pehe = np.mean(np.square((y_treated_pred-y_control_pred)-(y_treated_true-y_control_true)))
    ate = np.mean(np.abs((y_treated_pred - y_control_pred) - (y_treated_true - y_control_true)))

    return pehe,ate



def model_eval(model, dataloader, criterion ,eval_use_cuda=True):

    ipw_true, f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs,loss_all = transfer_data(model, dataloader, criterion, eval_use_cuda)

    pehe,ate = compute_pehe_ate(ipw_true, f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs)

    mse = mean_squared_error(f_outcome_true,f_outcome_outputs)

    print('PEHE: {:.4f}\tATE: {:.4f}\nRMSE: {:.4f}\n'.format(np.sqrt(pehe),ate, np.sqrt(mse)))

    return pehe, ate, mse, loss_all


# MAIN
if __name__ == '__main__':

    # ---------------------------------------- #
    # Parse input arguments

    treatment_option = 'vaso'
    gamma = '0.1'

    parser = argparse.ArgumentParser(description='Synthetic Dataset')

    parser.add_argument('--observation_window', type=int, default=12, required=True,
                        metavar='OW', help='observation window')

    parser.add_argument('--epochs', type=int, default=10, required=True,
                        metavar='EPOC', help='train epochs')

    parser.add_argument('--batch-size', type=int, default=64,
                        metavar='BS',help='batch size')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--l1', '--l1-reg-coef', default=1e-6, type=float,
                        metavar='L1', help='L1 reg coef')

    parser.add_argument('--resume', default=''.format(treatment_option), type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--save_model', default='checkpoints/mimic-6-7-{}.pt'.format(gamma), type=str, metavar='PATH',
                        help='path to save new checkpoint (default: none)')

    parser.add_argument('--cuda-device', default=0, type=int, metavar='N',
                        help='which GPU to use')

    parser.add_argument('--split_file', default='data_synthetic/data_syn_{}/train_test_split.csv'.format(gamma), type=str, metavar='PATH',
                        )

    # args = parser.parse_args()
    args = parser.parse_args(['--observation_window', '12', '--epochs', '10', '--batch-size', '64'])

    print("Settings:")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Observation window: {args.observation_window}")
    torch.manual_seed(666)

    try:
        train_test_split = np.loadtxt(args.split_file, delimiter=',', dtype=int)
    except Exception as e:
        print(f"Error loading train/test split file: {e}")
        raise

    train_iids = np.where(train_test_split==1)[0]
    val_iids = np.where(train_test_split == 2)[0]
    test_iids = np.where(train_test_split == 0)[0]

    # Datasets
    train_dataset = SyntheticDataset(train_iids, args.observation_window, treatment_option)
    val_dataset = SyntheticDataset(val_iids, args.observation_window, treatment_option)
    test_dataset = SyntheticDataset(test_iids, args.observation_window, treatment_option)

    # DataLoaders - removed CUDA generator
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    n_X_features, n_X_static_features, n_X_fr_types, n_classes = get_dim()
    # ---------------------------------------- #
    
    # Initialize model
    attn_model = 'concat2'
    n_Z_confounders = HIDDEN_SIZE

    model = LSTMModel(n_X_features, n_X_static_features, n_X_fr_types, n_Z_confounders,
                     attn_model, n_classes, args.observation_window,
                     args.batch_size, hidden_size=HIDDEN_SIZE)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    trainInitIPTW(train_loader, val_loader, test_loader,
                  model, epochs=args.epochs,
                  criterion=F.binary_cross_entropy_with_logits,
                  optimizer=optimizer,
                  l1_reg_coef=args.l1,
                  use_cuda=False,
                  save_model=args.save_model)
