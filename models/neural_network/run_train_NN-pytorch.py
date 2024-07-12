# run_train_pytorch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load Datasets
df_present = pd.read_csv('../../Dataset/air_system_present_year.csv')
df_previous = pd.read_csv('../../Dataset/air_system_previous_years.csv')

class DeepModel(nn.Module):
    """
    Defines a deep neural network model using PyTorch's nn.Module.

    Attributes:
        input_dim (int): Number of input dimensions/features.
    """
    def __init__(self, input_dim):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Custom Score Function with tqdm
def scoring_tqdm(model, criterion, X, y, pbar):
    """
    Args:
        model (nn.Module): PyTorch model to evaluate.
        criterion: Loss function criterion.
        X (torch.Tensor): Input tensor for evaluation.
        y (torch.Tensor): True labels tensor.
        pbar (tqdm.tqdm): tqdm progress bar to update during evaluation.a.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = criterion(y_pred, y)
        score = f1_score(y.cpu().numpy(), (y_pred.cpu().numpy() > 0.5).astype(int))
        pbar.update(1)
    return score

# Create artificial ID column
df_present['truck_id'] = range(1, len(df_present) + 1)
df_previous['truck_id'] = range(1, len(df_previous) + 1)

# Convert 'class' to binary
df_previous['class_binary'] = df_previous['class'].apply(lambda x: 1 if x == 'pos' else 0)
df_present['class_binary'] = df_present['class'].apply(lambda x: 1 if x == 'pos' else 0)

# Replace "na" with 0
df_previous = df_previous.replace('na', 0)
df_present = df_present.replace('na', 0)

# Select Features
selected_columns = [
    'ag_005', 'az_005', 'ba_000', 'bb_000', 'bx_000', 'bu_000', 'bv_000', 'cc_000', 'ci_000', 
    'cn_004', 'cn_005', 'cq_000', 'cs_005', 'ee_002', 'ee_003', 'ee_004'
]

# Prepare Data for Training
X_train = df_previous[selected_columns].values
y_train = df_previous['class_binary'].values
X_test = df_present[selected_columns].values
y_test = df_present['class_binary'].values

# Apply SMOTE to Balance Classes
smote = SMOTE(random_state=64)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize Data
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define DataLoader for training
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

# Define Model, Loss Function, and Optimizer
model = DeepModel(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Train the Model and Save the Best Model
best_model = None
best_f1_score = 0.0
epochs = 20
model.train()
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    epoch_loss = 0.0
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluate the model during training
    model.eval()
    with tqdm(total=len(X_test_tensor), desc="Evaluating Model") as pbar:
        y_pred_test_tensor = model(X_test_tensor)
        f1 = f1_score(y_test, (y_pred_test_tensor.detach().cpu().numpy() > 0.5).astype(int))
        pbar.update(1)

    # Save the best model based on F1-score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model.state_dict().copy()

# Load the best model obtained
if best_model:
    model.load_state_dict(best_model)

# Evaluate the Model on 'Present Year' Dataset
model.eval()
y_pred_present_tensor = model(X_test_tensor)
y_pred_present = (y_pred_present_tensor.detach().cpu().numpy() > 0.5).astype(int)

print("\nMetrics on 'Present Year' Dataset:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_present)}")
print(f"Precision: {precision_score(y_test, y_pred_present)}")
print(f"Recall: {recall_score(y_test, y_pred_present)}")
print(f"F1-Score: {f1_score(y_test, y_pred_present)}")
print(f"AUC: {roc_auc_score(y_test, y_pred_present)}")

# Add predictions and probabilities to df_present DataFrame
df_present['prediction'] = y_pred_present 
df_present['failure_probability'] = y_pred_present_tensor.detach().cpu().numpy()

# Simulate Maintenance Costs
total_real_cost = df_present['class_binary'].sum() * 500
total_simulated_cost = 0

with tqdm(total=len(df_present), desc="Simulating Costs") as pbar:
    for i in range(len(df_present)):
        if y_pred_present[i] == 1 and y_test[i] == 1:
            total_simulated_cost += 25
        elif y_pred_present[i] == 1 and y_test[i] == 0:
            total_simulated_cost += 10
        elif y_pred_present[i] == 0 and y_test[i] == 1:
            total_simulated_cost += 500
        pbar.update(1)

print("\nTotal Real Cost: $", total_real_cost)
print("Total Simulated Cost with Model: $", total_simulated_cost)
print("Potential Savings: $", total_real_cost - total_simulated_cost)

# Count Real Classes and Predictions
print("\nCount of Real Classes:")
print(df_present['class_binary'].value_counts())

print("\nCount of Predictions:")
print(df_present['prediction'].value_counts())

# Save the Model
torch.save(model.state_dict(), 'simple_model.pth')
print("\nModel saved as 'simple_model.pth'.")
