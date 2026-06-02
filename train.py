import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    # Combine input tensors and target tensors into PyTorch datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoader for training data
    # shuffle=True means the training samples are shuffled every epoch
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Create DataLoader for testing data
    # shuffle=False keeps the test data in the original order
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader


def train_step(model, dataloader, loss_fn, optimizer, device):
    # Set model to training mode
    model.train()

    train_loss = 0

    # Loop through each batch of data
    for X_batch, y_batch in dataloader:
        # Move data to CPU or GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass: make predictions
        prediction = model(X_batch)

        # Calculate loss between prediction and true target
        loss = loss_fn(prediction, y_batch)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation: calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate batch loss
        train_loss += loss.item()

    # Calculate average loss for this epoch
    train_loss = train_loss / len(dataloader)

    return train_loss


def test_step(model, dataloader, loss_fn, device):
    # Set model to evaluation mode
    model.eval()

    test_loss = 0

    # Disable gradient calculation during evaluation
    with torch.inference_mode():
        for X_batch, y_batch in dataloader:
            # Move data to CPU or GPU
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            prediction = model(X_batch)

            # Calculate loss
            loss = loss_fn(prediction, y_batch)

            # Accumulate batch loss
            test_loss += loss.item()

    # Calculate average loss for this epoch
    test_loss = test_loss / len(dataloader)

    return test_loss


def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device):
    # Store training and testing loss for each epoch
    results = {
        "train_loss": [],
        "test_loss": []
    }

    # Train and test the model for multiple epochs
    for epoch in range(epochs):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Save losses for later visualization
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f}"
    )

    return results
