def generate_triplets(images, labels):
  triplets = []
  for i in set(labels):
    same_class = np.where(labels == i)[0] # list
    diff_class = np.where(labels != i)[0]
    for anchor in same_class:
      positive = np.random.choice(same_class) # picks a random index from same class
      negative = np.random.choice(diff_class) # """" diff class
      triplets.append((anchor, positive, negative))
  return triplets



def support_set(images, labels, n_way, k_shot=5):
    support_images = []
    support_labels = []
    for digit in range(n_way):
        digit_indices = np.where(labels == digit)[0]
        selected_indices = np.random.choice(digit_indices, k_shot, replace=False)
        support_images.append(torch.tensor(images[selected_indices]))
        support_labels.extend([digit] * k_shot)
    return torch.cat(support_images), torch.tensor(support_labels)

def query_set(images, labels, n_query=100):
    query_indices = np.random.choice(range(len(images)), n_query, replace=False)
    return images[query_indices], labels[query_indices]
