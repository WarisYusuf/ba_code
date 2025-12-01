import sys, os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def reli_dia(konf, correct, uncert, n_bins=15, out_png="", uncert_name="Unc"):
    bins  = np.linspace(0, 1, n_bins+1)
    accs, konfs, counts, u_means = [], [], [], []
    for i in range(n_bins):
        min_b, max_b = bins[i], bins[i+1]
        mask = (konf >= min_b) & (konf < max_b if i < n_bins-1 else konf <= max_b)

        if np.any(mask):
            accs.append(correct[mask].mean())
            konfs.append(konf[mask].mean())
            counts.append(mask.sum())
            u_means.append(uncert[mask].mean())
        else:
            accs.append(np.nan)
            konfs.append((min_b+max_b)/2)
            counts.append(0)
            u_means.append(np.nan)

    b_avg = counts / max(sum(counts), 1)
    ece = np.nansum(b_avg * np.abs(np.array(accs) - np.array(konfs)))

    plt.figure(figsize=(10, 6))
    plt.plot([0,1],[0,1], '--',label ="Perfekt kalibriert")
    sc = plt.scatter(konfs, accs, c=u_means, cmap="viridis")
    plt.plot(konfs, accs, '-', label=f"ECE={ece:.3f}")
    cb = plt.colorbar(sc)
    cb.set_label(uncert_name)
    plt.xlabel("Konfidenz")
    plt.ylabel("Genauigkeit")
    plt.title("Kalibrierungsdiagram")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.clf()
    print(f"ECE={ece:.3f}")
    return ece


def confusionmatrix(df):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(df["y_true"], df["y_pred"]):
        cm[t, p] += 1
    return cm


def main(file_name):
    df = pd.read_csv(file_name)
    cm_ = confusionmatrix(df)
    pd.DataFrame(cm_).to_csv("confusion_matrix.csv")
    konf = df["conf_top1"].values
    corr = df["correct"].values
    u_an = df["aupnn_prob"].values
    m_an = df["mc_prob"].values


    reli_dia(konf, corr, uncert=u_an, n_bins=15, out_png="reliability_aupnn.png", uncert_name="AUPNN")
    reli_dia(konf, corr, uncert=m_an, n_bins=15, out_png="reliability_mc.png", uncert_name="MC")
    df_err = df[df["correct"] == 0].copy()
    cm_err = confusionmatrix(df_err)
    rows = []
    for i in range (10):
        for j in range(10):
            # skippe bei der Diagonalen und bei leeren Einträge
            if i == j or cm_err[i, j] == 0:
                continue
            a = cm_err[i, j]
            # filter genau nach den einzelnen Eintraegen
            fil = df_err[(i == df_err["y_true"]) & (j == df_err["y_pred"])]
            # y_true,y_pred,correct,logit_margin,aupnn_prob,aupnn_logit,mc_prob,mc_logit,mc_prob_top1,mc_logit_top1,aupnn_prob_top1,aupnn_logit_top1,mc_prob_top1_true,mc_logit_top1_true,aupnn_prob_top1_true,aupnn_logit_top1_true,conf_top1
            row = {
                "true": i,
                "pred": j, 
                "count_errors": a,
                "mean_margin": np.mean(fil["logit_margin"]),
                "aupnn_logit": np.mean(fil["aupnn_logit"]),
                "aupnn_prob": np.mean(fil["aupnn_prob"]),
                "mc_prob": np.mean(fil["mc_prob"]),
                "mc_logit": np.mean(fil["mc_logit"]),
                
                "mc_prob_top1": np.mean(fil["mc_prob_top1"]),
                "mc_logit_top1": np.mean(fil["mc_logit_top1"]),
                "aupnn_prob_top1": np.mean(fil["aupnn_prob_top1"]),
                "aupnn_logit_top1": np.mean(fil["aupnn_logit_top1"]),
                "mc_prob_top1_true": np.mean(fil["mc_prob_top1_true"]),
                "mc_logit_top1_true": np.mean(fil["mc_logit_top1_true"]),
                "aupnn_prob_top1_true": np.mean(fil["aupnn_prob_top1_true"]),
                "aupnn_logit_top1_true": np.mean(fil["aupnn_logit_top1_true"]),
            }
            rows.append(row)
    
    df_final = pd.DataFrame(rows).sort_values("count_errors", ascending=False)
    df_final.to_csv("confusions_errors.csv")
    
    margin = df["logit_margin"].values
    diff = np.abs(df["mc_prob"].values - df["aupnn_prob"].values)
    corr_ = np.corrcoef(margin, diff)[0,1]

    plt.figure(figsize=(10, 6))
    plt.scatter(margin, diff, label="Samples") 
    plt.xlabel("Logit-Margin (Top-1 - Top-2)")
    plt.ylabel("|MC - AUPNN| (im Softmax-Raum)")
    plt.title(f"Logit-Margin vs. |MC - AUPNN| r={corr_:.3f}")
    plt.tight_layout()
    plt.savefig("margin_vs_diff.png")
    plt.clf()


if __name__ == "__main__":
    main(sys.argv[1])
