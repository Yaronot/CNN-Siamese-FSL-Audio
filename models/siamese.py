# A basic CNN model for feature extraction

class Siamese(nn.Module):
  def __init__(self):
    super(Siamese, self).__init__()

    self.embedding = nn.Sequential(
        # input (*, 4, 64, 64)
        nn.Conv2d(4, 16, (5, 5), padding=2),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(16, 32, (3, 3), padding=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(32, 64, (3, 3), padding=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(64, 128, (3, 3), padding=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(128, 256, (3, 3), padding=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(0.5),
        nn.Linear(256, 128) # output is embedding representation of the spectrogram, not a classifier
    )

  def forward(self, x): # forward one image at a time
    return self.embedding(x)


