import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from prettytable import PrettyTable

# Function to evaluate the model
def evaluate_model(model, test_loader, tag):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Create a table
    table = PrettyTable()
    table.field_names = ["Metric", "Score"]
    # Adding rows with correct percentage formatting
    table.add_row(["Accuracy", f"{accuracy * 100:.2f}%"])
    table.add_row(["F1 Score", f"{f1 * 100:.2f}%"])
    table.add_row(["Precision", f"{precision * 100:.2f}%"])
    table.add_row(["Recall", f"{recall * 100:.2f}%"])
    print(table)

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tag": tag,
    }
    return result



def train_model(model, criterion, optimizer, train_loader, num_epochs):
    # Customize the bar format
    bar_format = '{desc}: {percentage:3.0f}%|{bar}|[{elapsed}{postfix}]'

    # Initialize a progress bar with customized bar format
    pbar = tqdm(range(num_epochs), desc="Training Progress", bar_format=bar_format)
    
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Update the progress bar with custom postfix
        pbar.set_postfix(Epoch=epoch+1, Loss=f"{running_loss:.2f}")
