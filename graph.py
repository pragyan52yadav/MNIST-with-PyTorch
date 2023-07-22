import numpy as np
import matplotlib.pyplot as plt

# epoch
epoch = np.array([0, 2, 5, 7, 10])

# LR = 0.001
avg_accu1 = np.array([0.2682, 0.9494, 0.9514, 0.9624, 0.968])

# LR = 0.009
avg_accu2 = np.array([0.2812, 0.8972, 0.906, 0.8938, 0.9092])

# LR = 1
avg_accu3 = np.array([0.1008, 0.1038, 0.0958, 0.1068, 0.0976])

# No of Epoch vs Avg Accuracy
plt.plot(epoch, avg_accu1, marker = 'x', label="LR=0.001")
plt.plot(epoch, avg_accu2, marker = 'x', label="LR=0.009")
plt.plot(epoch, avg_accu3, marker = 'x', label="LR=1")

plt.legend()
plt.title("No of Epoch vs Avg Accuracy")
plt.xlabel("No of Epoch")
plt.ylabel("Avg Accuracy")

plt.show()