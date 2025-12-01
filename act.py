import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 400)
relu = np.maximum(0,x)
alpha = 0.1
leaky_relu = np.where(x>0, x, alpha *x)


plt.figure(figsize=(8, 6))
plt.plot(x, relu, label="ReLU", color="blue")
plt.plot(x, leaky_relu, label="LeakyReLU", color="orange")
plt.xlabel("Eingabe")
plt.ylabel("Ausgabe")
plt.title("ReLU vs. LeakyReLU Aktivierungsfunktionen")
plt.ylim(-1, 5)
plt.legend()
plt.grid(True)
plt.savefig("leakyrelu_plot.png")
plt.clf()
