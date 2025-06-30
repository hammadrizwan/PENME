from imports import nn,torch




class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1=None):
        super(SiameseNetwork, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(input_size, hidden_size1),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size1, input_size),
        #     # nn.ReLU()

        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(input_size, hidden_size1),
        #     nn.LayerNorm(hidden_size1),  # Layer normalization replaces Batch normalization
        #     nn.ReLU(),
        #     nn.Linear(hidden_size1, input_size),
        #     # nn.LayerNorm(input_size)     # Layer normalization replaces Batch normalization
        # )
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),  # Batch normalization added
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.BatchNorm1d(input_size)     # Batch normalization added
        )
        # self.temperature = 0.01
    def forward_sequential(self, x):
        return self.fc(x)

    def forward(self, input1, input2):
        output1 = self.forward_sequential(input1)
        output2 = self.forward_sequential(input2)
        return output1, output2
        # return output1 / self.temperature, output2 / self.temperature



class SiameseClassificationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, num_classes=2):
        super(SiameseClassificationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(hidden_size1, hidden_size2),
            # nn.ReLU(),
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(input_size, num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward_sequential(self, x):
        return self.fc(x)

    def forward(self, input1, input2):

        output1 = self.forward_sequential(input1)
        output2 = self.forward_sequential(input2)
        output3 = self.sm(self.fc1(output1))


        return output1, output2, output3




class TwoLayerIdentityModel(nn.Module):
    def __init__(self, embedding_size):
        super(TwoLayerIdentityModel, self).__init__()
        # Define two linear layers
        self.layer1 = nn.Linear(embedding_size, embedding_size)
        self.layer2 = nn.Linear(embedding_size, embedding_size)
        
        # Initialize weights and biases for identity transformation
        nn.init.eye_(self.layer1.weight)  # Set weights to identity matrix
        nn.init.eye_(self.layer2.weight)
        nn.init.zeros_(self.layer1.bias)  # Set biases to zero
        nn.init.zeros_(self.layer2.bias)
        
    def forward(self, x,x2):
        x = self.layer1(x)
        x = self.layer2(x)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        return x,x2