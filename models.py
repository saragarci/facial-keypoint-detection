import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 4x4 square convolution kernel
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # the output Tensor for one image, will have the dimensions: (32, 221, 221)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (64, 108, 108)
        # after one pool layer, this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # third conv layer: 64 inputs, 128 outputs, 2x2 conv
        ## output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # the output tensor will have dimensions: (128, 53, 53)
        # after one pool layer, this becomes (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 2)
               
        # fourth conv layer: 128 inputs, 256 outputs, 1x1 conv
        ## output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output tensor will have dimensions: (256, 26, 26)
        # after one pool layer, this becomes (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # fifth conv layer: 256 inputs, 512 outputs, 1x1 conv
        ## output size = (W-F)/S +1 = (13-1)/1 +1 = 13
        # the output tensor will have dimensions: (512, 13, 13)
        # after one pool layer, this becomes (512, 6, 6)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        # 512 outputs * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1028)
        
        self.fc1_drop = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(1028, 256)
        
        self.fc2_drop = nn.Dropout(p=0.3)
        
        # finally, create 136 output channels (for the 136 keypoints)
        self.fc3 = nn.Linear(256, 136)
        

        
    def forward(self, x):
        # five conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)        
        x = self.fc3(x)
        
        # final output
        return x
