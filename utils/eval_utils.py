def evaluate_few_shot(model_path, support_images, support_labels, query_images, query_labels, n_way=10, k_shot=5, epochs=5, lr=0.0005, patience=7):
    model = Siamese()  # to extract the weights from the siamese model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.train()  # sets to train mode to be able to update weights

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_epoch = 0
    epochs_no_improve = 0
    best_predictions = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        support_features = model(support_images.to(device))
        query_features = model(query_images.to(device))

        distances = torch.cdist(query_features, support_features)
        _, indices = distances.topk(k_shot, largest=False, dim=1)
        support_labels = support_labels.to(device)
        nearest_labels = support_labels[indices]

        # Compute prototype for each class
        class_prototypes = torch.zeros(n_way, support_features.shape[1]).to(device)
        for i in range(n_way):
            class_mask = (support_labels == i).unsqueeze(1).to(device)
            class_prototypes[i] = (support_features * class_mask).sum(0) / class_mask.sum()

        # Compute logits as negative distances to prototypes
        logits = -torch.cdist(query_features, class_prototypes)

        # Compute loss
        loss = criterion(logits, query_labels.to(device))

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Make predictions
        _, predictions = torch.max(logits, 1)

        # Calculate accuracy
        accuracy = accuracy_score(query_labels.cpu(), predictions.cpu())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        # Early stopping check
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            # Save the best predictions
            best_predictions = predictions.cpu()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered. Best epoch: {best_epoch+1} with accuracy: {best_accuracy:.4f}")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    return best_accuracy, best_predictions, query_labels
