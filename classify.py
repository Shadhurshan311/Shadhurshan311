import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Load CSV ---
data = pd.read_csv("C:\\Users\\94772\\Desktop\\FYP\\Samples\\Data_crack.csv", header=None)

# --- Encode crack type labels ---
label_encoder = LabelEncoder()
data['CrackType'] = label_encoder.fit_transform(data['CrackType'])

# --- Split features and target ---
X = data[['Capacitance', 'Humidity', 'Temperature', 'Thickness']].values
y = data['CrackType'].values

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Normalize features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Convert to PyTorch tensors ---
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# --- Define model ---
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Train model ---
for epoch in range(300):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# --- Evaluate model ---
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy.item() * 100:.2f}%")
