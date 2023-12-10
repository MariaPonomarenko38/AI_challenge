import torch
import torch.nn as nn
import torch.optim as optim
from AI_challenge.model import ResidualNeuralNet
from torch.utils.data import Dataset, DataLoader
from AI_challenge.data import get_data_loader

INPUT_SIZE = 25
NUM_FC_LAYERS = 6
FC_SIZE = 120
NUM_BLOCKS = 4

model = ResidualNeuralNet(INPUT_SIZE, NUM_FC_LAYERS, FC_SIZE, NUM_BLOCKS)

train_loader, test_loader = get_data_loader(INPUT_SIZE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        print(outputs.shape)
        print(labels.shape)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
     
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    
    torch.save(model, 'resnet.pt')