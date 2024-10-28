class TripletLoss(nn.Module):
  def __init__(self, margin = 0.5):
    super(TripletLoss, self).__init__()
    self.margin = margin

  def forward(self, anchor, positive, negative):
    positive_dist = (anchor - positive).pow(2).sum(1) # calc positive distance
    negative_dist = (anchor - negative).pow(2).sum(1) # calc negative distance
    losses = F.relu(positive_dist - negative_dist + self.margin) # applying relu to the triple loss function - negatives become zeros and non negative stays the same
    return losses.mean()
