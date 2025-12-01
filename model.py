import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as func
import numpy as np
import torchvision.transforms as T
# from uncertainty_helper import uncertainty_helper


class ClassifierConv(Module):
    def __init__(self):
        super(ClassifierConv, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=16,
                      stride=(1, 1), 
                      kernel_size=(3,3),
                      padding=(1,1),
                      padding_mode="zeros"),
            nn.MaxPool2d(kernel_size=(2,2),
                        stride=(2,2)),
            nn.Conv2d(in_channels=16, 
                      out_channels=32,
                      stride=(1, 1), 
                      kernel_size=(3,3),
                      padding=(1,1),
                      padding_mode="zeros"),
            nn.MaxPool2d(kernel_size=(2,2),
                        stride=(2,2)),
                
            
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(64),
            nn.Linear(64, 10)
        )
        self.softmax = nn.Softmax(dim=1)
    

    # def forward(self, x):
    #     x = self.conv_layers(x)
    #     x = x.view(x.size(0), -1) # flatten
    #     x = self.mlp_layers(x)
    #     return x

   
    def predict_logits(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.mlp_layers(x) # Logits
        return x

    def forward(self, x, unc_factor=0.01, delta_in=None):
        device = x.device
        if delta_in is not None:
            delta = delta_in.to(device)
        else:
            delta = torch.ones_like(x, device=device) *unc_factor
   
        # blur = T.GaussianBlur(kernel_size=3)
        # x_blurr=blur(x)
        # delta = torch.abs(x - x_blurr) * unc_factor
        # fig, a = plt.subplots(3, 5, figsize=(12, 6))
        # for i in range(5):
        #     a[0, i].imshow(x[i].squeeze().cpu().detach().numpy(), cmap="gray")
        #     a[0, i].axis("off")
        #     # Original
        #     if i == 0:
        #         a[0, i].set_title("Original")
        #         a[1, i].set_title("Blurred")
        #         a[2, i].set_title("Eingabeunsicherheit")
        #     # Blurred
        #     a[1, i].imshow(x_blurr[i].squeeze().cpu().detach().numpy(), cmap="gray")
        #     a[1, i].axis("off")
        #     #DELTA
        #     a[2, i].imshow(delta[i].squeeze().cpu().detach().numpy(), cmap="hot")
        #     a[2, i].axis("off")
        # plt.tight_layout()
        # plt.savefig("delta_unc.png")
        # plt.close()

        x_out = x.clone()
        layer_names = []
        layer_names.append("Eingabe")
        uncertainties = []
        uncertainties.append(delta.mean().item())

        for i, layer in enumerate(self.conv_layers):
            x_out, delta, name, mean_uncertainty = self.uncertainty_helper(layer, x_out, delta, device, i)
            if name:
                layer_names.append(name)
            uncertainties.append(mean_uncertainty)
        x_out = x_out.view(x_out.size(0), -1)
        delta = delta.view(delta.size(0), -1)

        for i, layer in enumerate(self.mlp_layers):
            x_out, delta, name, mean_uncertainty = self.uncertainty_helper(layer, x_out, delta, device, i)
            if name:
                layer_names.append(name)
            uncertainties.append(mean_uncertainty)
        delta_logit = delta.clone()
        probs, delta, name, mu = self.uncertainty_helper(self.softmax, x_out, delta, device, "S")
        layer_names.append(name)
        uncertainties.append(mu)

        plt.figure(figsize=(10,6))
        plt.plot(layer_names, uncertainties,
                label="Durchschnittliche AUPNN im Batch", marker='o')
        plt.xlabel("Schicht (Layer)")
        plt.ylabel("Durchschnittliche AUPNN")
        plt.title("Unsicherheitsfortpflanzung durch das Modell")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("aupnn.png")
        plt.clf()

        probs_np = probs[0].detach().cpu().numpy()
        delta_np = delta_logit[0].detach().cpu().numpy()
        print("\n Vorhersage +- Unsicherheit pro Sample: ")
        for i, (p, u) in enumerate(zip(probs[0], delta[0])):
            print(f"Klasse {i}: {p.item():.2f} +- {u.item():.2f}")
        
        plt.figure(figsize=(10,6))
        plt.errorbar(np.arange(len(probs_np)), probs_np, yerr=delta_np,
                     label="Vorhersage +- Unsicherheit",
                     ecolor="orange", fmt='o', capsize=5)
        plt.xlabel("Klasse")
        plt.ylabel("Wahrscheinlichkeit")
        plt.title("Vorhersage mit AUPNN")
        plt.grid(True)
        plt.xticks(np.arange(len(probs_np)))
        plt.legend()
        plt.tight_layout()
        plt.savefig("prediction_uncertainty.png")
        plt.clf()
        return x_out, uncertainties, delta, delta_logit
        

    def uncertainty_helper(self, layer, x_out, delta, device, layer_index):
        layer_name = None
        if isinstance(layer, nn.Linear):
            weights = layer.weight.to(device)
            delta = torch.sqrt(torch.matmul(delta**2,(weights**2).T))
            x_out = layer(x_out)
            layer_name = f"Linear{layer_index}"
        elif isinstance(layer, nn.LeakyReLU):
            alpha = getattr(layer, "negative_slope", 0.01)
            f_prime = torch.where(x_out > 0, 1.0, alpha)
            delta = delta * f_prime
            x_out = layer(x_out)
            layer_name = f"LeakyReLU{layer_index}"

        elif isinstance(layer, nn.Conv2d):
            x_out = layer(x_out)
            delta = torch.sqrt(torch.nn.functional.conv2d(
                delta**2, 
                layer.weight**2,
                bias=None,
                stride=layer.stride,
                padding=layer.padding
            ))
            layer_name = f"Conv{layer_index}"
        
        elif isinstance(layer, nn.MaxPool2d):
            y, idx = func.max_pool2d(
                x_out, 
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                ceil_mode=layer.ceil_mode,
                return_indices=True
            ) # torch.Size([256, 16, 14, 14])

            mask = func.max_unpool2d(
                torch.ones_like(y), idx,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                output_size=x_out.shape[-2:]
            ) # an den Stellen, der idx 1 sonst 0 torch.Size([256, 16, 14, 14])
            delta = func.max_pool2d(
                delta * mask, 
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )
            x_out = y
            layer_name = f"MaxPool{layer_index}"

        elif isinstance(layer, nn.Softmax):
            x_out = layer(x_out) 
            s = x_out
            d = torch.zeros_like(delta)
            c = s.size(1)
            I = torch.eye(c, device=s.device, dtype=s.dtype).unsqueeze(0) # (1, C, C) Einheitsmatrix
            J = (s.unsqueeze(2) * (I - s.unsqueeze(1))) ** 2
            d = torch.sum( J * (delta**2).unsqueeze(1), dim=2)
            delta = torch.sqrt(torch.clamp(d, min=0.0))
            layer_name = "Softmax"
        return x_out, delta, layer_name, delta.mean().item()

    
