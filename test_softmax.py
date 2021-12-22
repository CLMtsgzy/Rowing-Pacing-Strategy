import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

data=autograd.Variable(torch.FloatTensor([1.0,2.0,3.0]))
log_softmax=F.log_softmax(data,dim=0)
print(log_softmax)

softmax=F.softmax(data,dim=0)
print(softmax)

print(1/(1+torch.exp(data)))