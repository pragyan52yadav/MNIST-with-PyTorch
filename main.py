import os
import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

print(t.__version__)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = t.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = t.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

# len(mnist_trainset)
# len(mnist_testset)

# the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28*28, 100) # input layer
        self.linear2 = nn.Linear(100, 50) # hidden layer
        self.final = nn.Linear(50, 10) # output layer
        self.relu = nn.ReLU() # piecewise linear function
    
    # convert + flatten
    def forward(self, img):
        x = img.view(-1, 28*28) # reshape the image for the model
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x

net = Net()

# loss function
cross_en_loss = nn.CrossEntropyLoss()
optimiser = t.optim.Adam(net.parameters(), lr=1) # e-1
epoch = 10

for epoch in range(epoch):
    net.train()
    
    for data in train_loader:
        x, y = data # x=features, y=targets
        optimiser.zero_grad() # set gradient to 0 before each loss calc
        output = net(x.view(-1, 28*28)) # pass in reshaped batch
        loss = cross_en_loss(output, y) # cal and grab the loss value
        loss.backward() # apply loss back through the network's parameters
        optimiser.step() # optimise weights to account for loss and gradients


# evaluating our dataset
correct = 0
total = 0
with t.no_grad():
    for data in test_loader:
        x, y = data
        output = net(x.view(-1, 784))
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct += 1
            total += 1
print(f"accuracy: {round(correct/total, 3)}")

# visualization
plt.imshow(x[3].view(28, 28))
plt.show()
print(t.argmax(net(x[3].view(-1, 784))[0]))