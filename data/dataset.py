class MyDataset(Dataset):
  def __init__(self, images, labels):
    self.images = images
    self.labels = labels

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    return self.images[index].permute(2,0,1), self.labels[index] # permutes the dimensions


class TripletDataset(Dataset):
    def __init__(self, spectrograms, triplets):
        self.spectrograms = spectrograms
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        anchor = self.spectrograms[anchor_idx].permute(2,0,1) # define anchor + premute dimensions
        positive = self.spectrograms[pos_idx].permute(2,0,1) # """"
        negative = self.spectrograms[neg_idx].permute(2,0,1) # """"
        return anchor, positive, negative
