import sys
import os
import math
import time 
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader 
from torch import optim as optim
from torch import nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from dataset import MNISTDataset
from model import ClassifierConv
import random


# globale variablen, hyperparameter
fixed_seed = 42
torch.manual_seed(fixed_seed)
torch.backends.cudnn.deterministic =False #Reihenfolge wird die Eregbnisse immer gleich, soll auf False
torch.backends.cudnn.benchmark= False #benchmark sollte auf true sein,  misst beim ersten Batch verschiedene Kerneös und wählt den schnellsten 

num_workers = 4
num_epochs = 10
subset_factor = 1
testing_factor = 0.2
batch_size = 256
learning_rate = 0.001
adam_eps = 1e-08
adam_betas = (0.9, 0.999)

file_name = 'MNISTDataset.csv'



#sychronisiert cuda
def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def runtime_aupnn(model, x, sigma=0.1, delta_in= None):
    device= x.device
    cuda_sync(device)
    t0 = time.perf_counter() * 1000.0
    _ = model(x, unc_factor= sigma, delta_in=delta_in)
    cuda_sync(device)
    t1 = time.perf_counter() * 1000.0
    return t1 - t0


@torch.no_grad()
def runtime_mc(model, x, Tn= 64, sigma=0.1, delta_in= None):
    device= x.device
    cuda_sync(device)
    t0 = time.perf_counter() * 1000.0
    _ = mc_aleatoric(model,x, Tn=Tn, unc_factor= sigma, device=device, delta_in=delta_in)
    cuda_sync(device)
    t1 = time.perf_counter() * 1000.0
    return t1 - t0

@torch.no_grad()
def runtime(model, testdl, device, sigma=0.10, Tn=2000):
    mc_l = []
    aupnn_l = []
    model.eval()
    x,_ = next(iter(testdl))
    x = x.to(device)
    delta_in= torch.ones_like(x)*sigma
    ########aupnn 
    for i in range(20):
        times_aupnn = runtime_aupnn(model,x, delta_in=delta_in)
        aupnn_l.append(times_aupnn)
    mean_aupnn = np.mean(aupnn_l)

    #### MC
    for t in range( 1, Tn+1):
        times_mc= runtime_mc(model,x, Tn=t, delta_in=delta_in)
        mc_l.append(times_mc)

    #############PLOT
    plt.figure(figsize=(8,6))
    plt.plot(range(len(mc_l)), mc_l, label="MC (ms)", marker='o')
    plt.axhline(mean_aupnn, label=f"AUPNN (ms) = {mean_aupnn:.1f}", linestyle='--', color='red')
    plt.xlabel("Tn (MC-Samples)")
    plt.ylabel("Zeit [ms]")
    plt.title("Laufzeitvergleich: MC vs. AUPNN")
    plt.grid(True)
    plt.legend()
    plt.savefig("laufzeit_mc_aupnn.png")
    plt.clf()

@torch.no_grad()
def res_csv(model, dataloader, device, out_csv, Tn=20, sigma =0.10, use_blur=True):
    model.eval()
    rows =[]

    for x, y_true in dataloader:
        x = x.to(device)
        y_true = y_true.to(device)
        logits= model.predict_logits(x)
        probs  = func.softmax(logits, dim=1)
        top2 = torch.topk(logits, k=2, dim=1)
        y_pred= top2.indices[:, 0]
        logit_margin= top2.values[:, 0] - top2.values[:, 1]
        correct = (y_pred == y_true).int()

        if use_blur:
            x_blur = T.GaussianBlur(kernel_size=3, sigma=1.0)
            delta_in = torch.abs(x- x_blur(x)) *sigma
        else:
            delta_in = torch.ones_like(x) *sigma 


        #########MC
        (mc_var_probs, mc_std_probs), (mc_var_logits, mc_std_logits) = mc_aleatoric(model,
                                                                                    x, Tn=Tn, unc_factor=sigma,
                                                                                    device=device, delta_in=delta_in)

        idx = y_pred.view(-1, 1)
        idx_true = y_true.view(-1,1)
        conf_top1 = torch.gather(probs, 1, idx).squeeze(1)

       
        mc_prob_top1 = torch.sqrt(torch.gather(mc_var_probs, 1, idx).squeeze(1))# unsicherheiten für die vorhergesagten Klassen
        mc_logit_top1 = torch.sqrt(torch.gather(mc_var_logits, 1, idx).squeeze(1))# unsicherheiten für die vorhergesagten Klassen
        mc_prob_top1_true = torch.sqrt(torch.gather(mc_var_probs, 1, idx_true).squeeze(1))# unsicherheiten für die vorhergesagten Klassen
        mc_logit_top1_true = torch.sqrt(torch.gather(mc_var_logits, 1, idx_true).squeeze(1))

        ########AUPNN 
        _, _, delta_probs, delta_logits = model(x, unc_factor=sigma, delta_in=delta_in) 

        aupnn_prob = torch.sqrt((delta_probs**2).sum(dim=1))
        aupnn_logit = torch.sqrt((delta_logits**2).sum(dim=1))

        aupnn_prob_top1 = torch.sqrt(torch.gather(delta_probs, 1, idx).squeeze(1))# unsicherheiten für die vorhergesagten Klassen
        aupnn_logit_top1 = torch.sqrt(torch.gather(delta_logits, 1, idx).squeeze(1))# unsicherheiten für die vorhergesagten Klassen
        aupnn_prob_top1_true = torch.sqrt(torch.gather(delta_probs, 1, idx_true).squeeze(1))# unsicherheiten für die vorhergesagten Klassen
        aupnn_logit_top1_true = torch.sqrt(torch.gather(delta_logits, 1, idx_true).squeeze(1))
        
        
        for i in range(x.size(0)):
            rows.append({
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "correct": int(correct[i]),
                "logit_margin": float(logit_margin[i]),
                "aupnn_prob": float(aupnn_prob[i]),
                "aupnn_logit": float(aupnn_logit[i]),
                "mc_prob": float(mc_std_probs[i]),
                "mc_logit": float(mc_std_logits[i]),

                "mc_prob_top1": float(mc_prob_top1[i]),
                "mc_logit_top1": float(mc_logit_top1[i]),
                "aupnn_prob_top1": float(aupnn_prob_top1[i]),
                "aupnn_logit_top1": float(aupnn_logit_top1[i]),
                "mc_prob_top1_true": float(mc_prob_top1_true[i]),
                "mc_logit_top1_true": float(mc_logit_top1_true[i]),
                "aupnn_prob_top1_true": float(aupnn_prob_top1_true[i]),
                "aupnn_logit_top1_true": float(aupnn_logit_top1_true[i]),
                "conf_top1": float(conf_top1[i])


            })

    print(rows)
    pd.DataFrame(rows).to_csv(out_csv)

@torch.no_grad()
def stufen_delta(model, test_dl, device, sigmas, use_blur=True, a=""):
    model.eval()
    #ich will alle testdaten aufeinmal, net in batches
    xs, ys =[],[]
    for x_b, y_b in test_dl:
        xs.append(x_b)
        ys.append(y_b)
    x_all = torch.cat(xs, dim=0).to(device)
    y_all = torch.cat(ys, dim=0).to(device)

    acc_list = []
    mc_prob_list, mc_logit_list = [],[]
    an_prob_list, an_logit_list = [],[]
    corr_prob_list, corr_logit_list = [],[]
    rmse_prob_list, rmse_logit_list = [],[]
    delta_means = []

    for sigma in sigmas:
        if use_blur:
            x_blur = T.GaussianBlur(kernel_size=3, sigma=1.0)
            delta_in = torch.abs(x_all- x_blur(x_all)) *sigma
            delta_mean = delta_in.detach().cpu().abs().mean().item()
            delta_means.append(delta_mean)
        else:
            delta_in = torch.ones_like(x_all) *sigma 

        ######### AUPNN 
        _, _, delta_probs, delta_logits = model(x_all, unc_factor=sigma, delta_in=delta_in)
        an_std_prob = torch.sqrt((delta_probs**2).sum(dim=1))
        an_std_logit = torch.sqrt((delta_logits**2).sum(dim=1))
        an_prob_list.append(an_std_prob.mean().item())
        an_logit_list.append(an_std_logit.mean().item())

        ############# MC
        (_, mc_std_probs), (_, mc_std_logits) = mc_aleatoric(model, x_all, Tn = 300,
                                                             unc_factor=sigma, device=device,
                                                             delta_in=delta_in)

        mc_prob_list.append(mc_std_probs.mean().item())
        mc_logit_list.append(mc_std_logits.mean().item())
        
        ############### RMSE, CORR
        a_log_np = an_std_logit.detach().cpu().numpy()
        a_prob_np = an_std_prob.detach().cpu().numpy()
        m_log_np = mc_std_logits.detach().cpu().numpy()
        m_prob_np = mc_std_probs.detach().cpu().numpy()
        
        corr_prob_list.append(corr(a_prob_np, m_prob_np))
        corr_logit_list.append(corr(a_log_np, m_log_np))
        rmse_prob_list.append(rmse(a_prob_np, m_prob_np))
        rmse_logit_list.append(rmse(a_log_np, m_log_np))

        ######### ACC
        acc_sigma = 0.0
        for i in range(8):
            eps = torch.randn_like(x_all)
            x_noisy = x_all + delta_in *eps
            z_noisy = model.predict_logits(x_noisy)
            pred_noisy = z_noisy.argmax(dim=1)
            acc_sigma += (pred_noisy == y_all).float().mean().item()
        acc_sigma /= 8
        acc_list.append(acc_sigma)

    ##### PLOTS
    if use_blur:
        xs_np = delta_means 
    else:
        xs_np =sigmas

    plt.figure(figsize=(8,6))
    plt.plot(xs_np, acc_list, marker="o")
    plt.xlabel("Eingabeunsicherheit")
    plt.ylabel("Genauigkeit")
    plt.title("Genauigkeit vs. Eingabeunsicherheit")
    plt.grid(True)
    plt.savefig(f"stufen_acc_sigma_{a}.png")
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.plot(xs_np, mc_prob_list, label ="MC", marker="o",color ="blue")
    plt.plot(xs_np, an_prob_list, label ="AUPNN", marker="o",color ="orange")
    plt.xlabel("Eingabeunsicherheit")
    plt.ylabel("MC vs. AUPNN (Softmax-Raum)")
    plt.title("Ausgabeunsicherheit vs. Eingabeunsicherheit")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"stufen_delta_softmax_{a}.png")
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.plot(xs_np, mc_logit_list, label ="MC", marker="o",color ="blue")
    plt.plot(xs_np, an_logit_list, label ="AUPNN", marker="o",color ="orange")
    plt.xlabel("Eingabeunsicherheit")
    plt.ylabel("MC vs. AUPNN (Logit-Raum)")
    plt.title("Ausgabeunsicherheit vs. Eingabeunsicherheit")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"stufen_delta_logit_{a}.png")
    plt.clf()

    print("jooooooooooooooo")


def loss_uncertainty(prob, delta, target):
    y_onehot = func.one_hot(target, 10) # prob hat dim 256, 10 und target 256, selbe Dim bringen
    dL = prob - y_onehot # Das wird aber minus wegen minus 1 
    delta_L = torch.sqrt(((dL * delta)**2).sum(dim=1))# summiere über die 10 Klassen 
    return delta_L


def corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.corrcoef(x,y)[0, 1]

def rmse(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.mean((x-y)**2))

#####Vergleich zwischen AUPNN und MC, 2 plots
def plot_mc_aupnn(a, m, out_prefix, title_prefix, log=False):
    a = np.asarray(a).ravel()
    m = np.asarray(m).ravel()

    corr_ = corr(a, m)
    rmse_ = rmse(a, m)
    min_ = min(a.min(), m.min())
    max_ = max(a.max(), m.max())
    g = np.linspace(min_, max_, 200)

    cvals = np.abs(m-a)


    plt.figure(figsize=(10,6))
    sc = plt.scatter(a, m, c=cvals, cmap="viridis")
    plt.plot(g, g, label ="x = y", linewidth = 2, color ="orange")
    plt.xlabel("AUPNN")
    plt.ylabel("MC")
    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(f"AUPNN vs. MC {title_prefix} r={corr_:.3f} RMSE={rmse_:.4f}")
    plt.grid(True)
    cb = plt.colorbar(sc)
    cb.set_label("|MC - AUPNN|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_scatter.png")
    plt.clf()

    #######
    sort = np.argsort(a)
    a_sorted = a[sort]
    m_sorted = m[sort]
    xi = np.arange(len(a_sorted))
    plt.figure(figsize=(10,6))
    for i in range(len(xi)):
        plt.plot([xi[i], xi[i]], [m_sorted[i], a_sorted[i]], linewidth=0.8, alpha=0.6, color="gray")
    plt.scatter(xi, m_sorted, label= "MC", marker="o")
    plt.scatter(xi, a_sorted, label= "AUPNN", marker="^")

    plt.xlabel("Datenpunkte")
    plt.ylabel("Ausgabeunsicherheit")
    if log:
        plt.yscale("log")
    plt.title(f"Gepaart: AUPNN vs. MC {title_prefix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_paired.png")
    plt.clf()

    return corr_, rmse_

def plot_per_class_corr(aupnn, mc_var, out_path, title, log=False):
    m = np.sqrt(mc_var)
    c = aupnn.shape[1]
    corrs = []
    for i in range(c):
        corrs.append(corr(aupnn[:, i], m[:, i]))
    plt.figure(figsize=(10,4))
    plt.bar(np.arange(c), corrs)
    plt.xlabel("Klasse")
    if log:
        plt.yscale("log")
    plt.ylabel("Korrelation zwischen AUPNN und MC")
    plt.ylim(-0.25, 1.0)
    plt.title(title)
    plt.savefig(out_path)
    plt.clf()
    return corrs

###### MC ALS vergleichsmethode
def mc_aleatoric(model, x, Tn=5000, unc_factor=0.01, device=None, delta_in =None):
    if device is None:
        device = x.device
    gen = torch.Generator(device=device).manual_seed(fixed_seed)
    model.eval()
    if delta_in is not None:
        delta_in = delta_in.to(device)
    else:
        delta_in = torch.ones_like(x) *unc_factor

    with torch.no_grad():
        probs_T, logits_T = [], []
        for i in range(Tn):
            noise = torch.randn(x.shape, device=device, generator=gen) # dim: (batch, 1, 28, 28)
            x_noisy = x + delta_in * noise
            z = model.predict_logits(x_noisy.to(device)) # dim: (batch,klasses)
            s = func.softmax(z, dim=1) # dim: (batch, klasse)
            logits_T.append(z)
            probs_T.append(s)

        logits_T = torch.stack(logits_T, dim=0)
        probs_T = torch.stack(probs_T, dim=0) #Dim erweitern mit Tn

        var_logits = logits_T.var(dim=0, unbiased=False) # (batch, class)
        var_probs = probs_T.var(dim=0, unbiased=False) # (batch, class)

        std_logits = torch.sqrt(var_logits.sum(dim=1)) # (batch, class)
        std_probs = torch.sqrt(var_probs.sum(dim=1)) # (batch, class)

    return (var_probs, std_probs), (var_logits, std_logits) 


####DONE bis hier 

def plot_loss(loss_per_epoch_training, loss_per_epoch_test):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_per_epoch_training) +1), loss_per_epoch_training, label="Training Loss", marker="o")
    plt.plot(range(1, len(loss_per_epoch_test) +1), loss_per_epoch_test, label="Testing Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_loss.png")
    plt.clf()

def plot_accuracies(accuracies_per_epoch_training, accuracies_per_epoch_test):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies_per_epoch_training) +1), accuracies_per_epoch_training, label="Training Accuracy", marker="o")
    plt.plot(range(1, len(accuracies_per_epoch_test) +1), accuracies_per_epoch_test, label="Testing Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_accuracies.png")
    plt.clf()

def prepare_data(sample_factor, testset_factor):
    pd_data = pd.read_csv(file_name, header=None, delimiter=',')
    pd_data = pd_data.replace(['-nan', 'nan', 'inf', '-inf'], np.nan)
    pd_data = pd_data.dropna()
    pd_data = pd_data.astype(float)
    pd_data = pd_data.sample(frac=sample_factor, random_state=fixed_seed).reset_index(drop=True)
    train, test = train_test_split(pd_data, test_size=testset_factor)
    training_set = MNISTDataset(train, sample_factor)
    test_set = MNISTDataset(test, sample_factor)
    training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return training_dataloader, test_dataloader


def start_training(training_dl, test_dl):
    model = ClassifierConv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=adam_eps, betas=adam_betas)
    loss_function = nn.CrossEntropyLoss().to(device)

    loss_per_epoch_training = []
    loss_per_epoch_test = []
    accuracies_per_epoch_training = []
    accuracies_per_epoch_test = []

    uncert_train = []
    uncert_train_2 = []
    uncert_test = []
    uncert_test_2 = []
    mc_u_over_epochs = []

    for epoch in range(num_epochs):
        model.train()
        loss_training = 0
        correct_classified_training = 0
        total_classified_training = 0 
        train_dataloader_progress = tqdm(training_dl, desc="Training")
        all_uncertainties_train = []

        for batch in train_dataloader_progress:
            input_features, labels = batch 
            input_features = input_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # model_output = model(input_features)
            model_output, uncert, delta_probs, logit_delta = model(input_features) # uncert ist [8]
            print(uncert)
            last_uncert = uncert[-1]

            all_uncertainties_train.append(last_uncert)
            probabilities = func.softmax(model_output, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)
            correct_classified_training += (predicted_labels == labels).sum().item() # nur die correcten aufsummieren
            total_classified_training += labels.size(0) #anzahl der elemente in der ersten dim
            loss = loss_function(model_output, labels)
            loss.backward()
            optimizer.step()
            loss_training += loss.item() # aufsumierter Loss aller Batches, bu
            train_dataloader_progress.set_description(f'Training -> Epoch {epoch + 1}/{num_epochs}')

        loss_training /= len(training_dl)
        loss_per_epoch_training.append(loss_training)
        accuracies_per_epoch_training.append(correct_classified_training/ total_classified_training)
        uncert_train.append(np.mean(all_uncertainties_train))

        model.eval()
        loss_testing = 0 
        correct_classified_testing = 0
        total_classified_testing = 0
        test_dataloader_progress = tqdm(test_dl, desc='Testing')
        all_uncertainties_test = []

        with torch.no_grad():
            for batch in test_dataloader_progress:
                input_features, labels = batch
                input_features = input_features.to(device)
                labels = labels.to(device)
                model_output, uncert, delta_probs, logit_uncert = model(input_features)
                last_test_unc = uncert[-1]
                all_uncertainties_test.append(last_test_unc)
                probabilities = func.softmax(model_output, dim=1)
                _, predicted_labels = torch.max(probabilities, 1)
                correct_classified_testing += (predicted_labels == labels).sum().item()
                total_classified_testing += labels.size(0)
                loss = loss_function(model_output, labels)
                loss_testing += loss.item()
                test_dataloader_progress.set_description(f'Testing -> Epoch {epoch + 1}/{num_epochs}')
        
        loss_testing /= len(test_dl)
        loss_per_epoch_test.append(loss_testing)
        accuracy_testing = correct_classified_testing / total_classified_testing
        accuracies_per_epoch_test.append(accuracy_testing)
        uncert_test.append(np.mean(all_uncertainties_test))
        uncert_test_2.append(all_uncertainties_test)


################### MC vergleich
        # res_csv(model, test_dl, device=device, use_blur=False, out_csv=f"results_epoch_{epoch+1}.csv")
        x_t, _ = next(iter(test_dl))
        x_t = x_t.to(device)
        unc_factor = 0.01
        Tn = 2000
        (mc_var_probs, mc_std_probs), (mc_var_logits, mc_std_logits) = mc_aleatoric(model, x_t,
                                                                                    Tn=Tn, unc_factor=unc_factor, device=device)

        ### JE NACH dem mit oder ohne blurring
        _, _, delta_probs_t, delta_logits_t = model(x_t, unc_factor=unc_factor)
        # blur = T.GaussianBlur(kernel_size=3)
        # x_blurr = blur(x_t)
        # delta_in = torch.abs(x_t - x_blurr) *unc_factor
        # print(epoch,delta_in)


        aupnn_std_probs = torch.sqrt((delta_probs_t**2).sum(dim=1))
        aupnn_std_logits = torch.sqrt((delta_logits_t**2).sum(dim=1))

        a_prob = aupnn_std_probs.detach().cpu().numpy()
        m_prob = mc_std_probs.detach().cpu().numpy()
        a_log = aupnn_std_logits.detach().cpu().numpy()
        m_log = mc_std_logits.detach().cpu().numpy()


        df = pd.DataFrame({
            "Sample": np.arange(1, len(a_prob) + 1),
            "aupnn_prob": a_prob,
            "mc_prob": m_prob,
            "aupnn_logit": a_log,
            "mc_logit": m_log,


        })
        df.to_csv(f"mc_vs_aupnn_epoch__{epoch+1}.csv", index=False)
        plot_mc_aupnn(
            a_prob, m_prob,
            out_prefix=f"mc_vs_aupnn_PROB_epoch_{epoch+1}",
            title_prefix="(Softmax-Raum)"
        )
        
        plot_mc_aupnn(
            a_prob, m_prob,
            out_prefix=f"mc_vs_aupnn_PROB_epoch_{epoch+1}_log",
            title_prefix="(Softmax-Raum)",
            log=True
        )
        plot_mc_aupnn(
            a_log, m_log,
            out_prefix=f"mc_vs_aupnn_LOGIT_epoch_{epoch+1}",
            title_prefix="(Logit-Raum)"
        )
        #####KORRELATION
        plot_per_class_corr(
            delta_probs_t.detach().cpu().numpy(),
            mc_var_probs.detach().cpu().numpy(),
            f"per_class_corr_PROB_epoch_{epoch+1}.png",
            "Korrelation pro Klasse (Softmax-Raum)"
        )
        plot_per_class_corr(
            delta_probs_t.detach().cpu().numpy(),
            mc_var_probs.detach().cpu().numpy(),
            f"per_class_corr_PROB_epoch_{epoch+1}_log.png",
            "Korrelation pro Klasse (Softmax-Raum)",
            log=True
        )

        plot_per_class_corr(
            delta_logits_t.detach().cpu().numpy(),
            mc_var_logits.detach().cpu().numpy(),
            f"per_class_corr_LOGIT_epoch_{epoch+1}.png",
            "Korrelation pro Klasse (Logit-Raum)"
        )

        mc_u_over_epochs.append(mc_std_logits.mean().item())
        print(f"Epoch {epoch + 1 } /{num_epochs}")
        print(f"\t - Train Accuracy: {accuracies_per_epoch_training[-1]:.4f}")
        print(f"\t - Train Loss:     {loss_training:.4f}")
        print(f"\t - Test Accuracy: {accuracy_testing:.4f}")
        print(f"\t - Test Loss: {loss_testing:.4f}")

        
    # stufen_delta(
    #     model, test_dl, device,
    #     sigmas=np.round(np.arange(0.0, 30, 0.5), 2),
    #     use_blur=True, a="blur"
    # )
    print("Training complete.")
    print("Training loss: ", loss_per_epoch_training)
    print("Test loss: ", loss_per_epoch_test)
    print("Training uncertainty: ", uncert_train)
    print("Test uncertainty: ", uncert_test)

    plot_mc_over_epochs(mc_u_over_epochs)

    return accuracies_per_epoch_training, \
        accuracies_per_epoch_test, \
        loss_per_epoch_training, \
        loss_per_epoch_test, \
        uncert_train, \
        uncert_test
    

def plot_uncertainty(unc_train, unc_test=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(unc_train) +1), unc_train, label="Training AUPNN", marker="o")
    if unc_test is not None:
        plt.plot(range(1, len(unc_test) +1), unc_test, label="Testing AUPNN", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("AUPNN")
    plt.title("Training and Testing AUPNN Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_uncertainty.png")
    plt.clf()


def plot_mc_over_epochs(mc_u):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mc_u) +1), mc_u, label="MC", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("MC")
    plt.title("MC über die Epochen")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_mc.png")
    plt.clf()




if __name__ == '__main__':
    training_dl, test_dl = prepare_data(subset_factor, testing_factor)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # model = ClassifierConv().to(device)
    # res_csv(model, test_dl, device=device, use_blur=False, out_csv=f"results_epoch_t.csv")
    # stufen_delta(
    #     model, test_dl, device,
    #     sigmas=np.arange(0.0, 2.51, 0.5),
    #     use_blur=True
    # )
    results = start_training(training_dl, test_dl)
    # runtime(model, test_dl, device,sigma=0.10, Tn=500)
    accuracies_training = results[0]
    accuracies_test = results[1]
    training_losses = results[2]
    test_losses = results[3]
    training_unc = results[4]
    test_unc = results[5]
    plot_loss(training_losses, test_losses)
    plot_accuracies(accuracies_training, accuracies_test)
    plot_uncertainty(training_unc, test_unc)
