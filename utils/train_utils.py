epochs = 5  # Or however many epochs you want
model = Siamese()
criterion = TripletLoss(margin=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

save_dir = '/content/drive/MyDrive/siamese_model'
os.makedirs(save_dir, exist_ok=True)

# set conditions for early stopping
best_loss = float('inf')
patience = 0
max_patience = 1

train_losses, val_losses = [], []

for epoch in range(epochs):
    # train
    model.train()
    running_loss = 0

    for anchor, positive, negative in tqdm.tqdm(S_train_digits_triplets_dataloader, desc='train'):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(S_train_digits_triplets_dataloader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} | train loss: {epoch_loss:.4f}")

    # Check if loss is increasing
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience = 0
        # Save the best model
        best_model_path = os.path.join(save_dir, 'siamese_model_best.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved to {best_model_path}")
    else:
        patience += 1

    if patience >= max_patience:
        print(f"Early stopping triggered. Loss has not improved for {max_patience} consecutive epochs.")
        break

# Save the final model
final_save_path = os.path.join(save_dir, 'siamese_model_final.pth')
torch.save(model.state_dict(), final_save_path)
print(f"Final model weights saved to {final_save_path}")
