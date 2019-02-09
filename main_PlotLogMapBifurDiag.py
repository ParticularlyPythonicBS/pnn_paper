from LogisticMap import BifurcationDiagram
import numpy as np

# Plot the bifurcation diagram of the logistic map (Fig. 3a of the paper)

N = 1500
x0 = 0.5
rs = np.linspace(2.5, 4, 1000)

BifurcationDiagram(rs, x0, N)
