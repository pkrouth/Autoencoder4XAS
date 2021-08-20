from ..utils.imports import *
class autoencoder(nn.Module):
    '''Auto Encoder with 3 Linear Layer architecture'''
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 10 * 10), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
## Flatten and UnFlatten Layers 
class Flatten(nn.Module):
    def __init__(self, full=False):
        super(Flatten, self).__init__()
        self.full = full
    def forward(self,x):
        return x.view(-1) if self.full else x.view(x.size(0),-1)

class UnFlatten(nn.Module):
    def forward(self,x):
        return x.view(x.size(0),32,-1)
    
class Conv1Dautoencoder(nn.Module):
    """
    bs=32
    test_X = torch.rand((bs,1,100))
    test_X.shape,
    conv_ = nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1)(test_X)
    print("Conv Layer#1: ",test_X.shape,conv_.shape)
    conv_ = nn.AvgPool1d(kernel_size=3, stride=1)(conv_)
    print("AvgPool Layer#1: ",test_X.shape, conv_.shape)
    conv_ = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1)(conv_)
    print("Conv Layer#2: ",test_X.shape,conv_.shape)
    conv_ = nn.AvgPool1d(kernel_size=3, stride=2)(conv_)
    print("AvgPool Layer#2: ",test_X.shape, conv_.shape)
    conv_ = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1)(conv_)
    print("Conv Layer#3: ",test_X.shape,conv_.shape)
    conv_ = nn.AvgPool1d(kernel_size=3, stride=3)(conv_)
    print("AvgPool Layer#3: ",test_X.shape,conv_.shape)
    conv_ = Flatten(full=False)(conv_)
    print("Flattened Layer:", test_X.shape, conv_.shape)
    lin_ = nn.Linear(96,10)(conv_)
    print("Linear Layer:", test_X.shape, lin_.shape)
    >>> Conv Layer#1:  torch.Size([32, 1, 100]) torch.Size([32, 8, 100])
        AvgPool Layer#1:  torch.Size([32, 1, 100]) torch.Size([32, 8, 98])
        Conv Layer#2:  torch.Size([32, 1, 100]) torch.Size([32, 16, 50])
        AvgPool Layer#2:  torch.Size([32, 1, 100]) torch.Size([32, 16, 24])
        Conv Layer#3:  torch.Size([32, 1, 100]) torch.Size([32, 32, 11])
        AvgPool Layer#3:  torch.Size([32, 1, 100]) torch.Size([32, 32, 3])
        Flattened Layer: torch.Size([32, 1, 100]) torch.Size([32, 96])
        Linear Layer: torch.Size([32, 1, 100]) torch.Size([32, 10])
    """
    def __init__(self, latent_size=10):
        super(Conv1Dautoencoder, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=2), # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=1), # Output: #bs, 8, 98
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1),  # Output: b, 16, 50
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1),  # Output: b, 32, 11
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=3),  # Output: b, 32, 3
            Flatten(full=False),
            nn.Linear(96,self.latent_size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 32*3),
            UnFlatten(),
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=2), # b, 16, 10
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 10, stride=2, padding=0),  # b, 1, 100
            nn.Tanh()
            
            
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
