import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
#import svm 
#import decision tree algorithms
# import MLP


# Step 1: Load Data
data = pd.read_csv('C:\\Users\\94772\\Desktop\\FYP\\Samples\\crack_data.csv')

# Step 2: Encode Labels
label_encoder = LabelEncoder()
data['CrackType'] = label_encoder.fit_transform(data['CrackType'])  # Hairline=0, Mild=1, Severe=2

# Step 3: Split Features and Target
X = data[['Capacitance', 'Temperature', 'Humidity', 'Thickness']].values
y = data['CrackType'].values

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 7: Build Classifier
mlp_model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)  # 3 output classes
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01)

# Step 8: Training Loop
for epoch in range(300):
    mlp_model.train()
    outputs = mlp_model(X_train)
    loss = loss_fn(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# Step 9: Evaluate
mlp_model.eval()
with torch.no_grad():
    predictions = mlp_model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)

    accuracy = (predicted_classes == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy.item() * 100:.2f}%")

# Optional: Print predictions
predicted_labels = label_encoder.inverse_transform(predicted_classes.numpy())
actual_labels = label_encoder.inverse_transform(y_test.numpy())
for actual, predicted in zip(actual_labels, predicted_labels):
    print(f"Actual: {actual}, Predicted: {predicted}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Run prediction (you already have this after training)
mlp_model.eval()
with torch.no_grad():
    y_pred_mlp = mlp_model(X_test)
    predicted_classes = torch.argmax(y_pred_mlp, dim=1)

# Confusion Matrix
cm = confusion_matrix(y_test.numpy(), predicted_classes.numpy())
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_).plot(cmap='Oranges')
plt.title("MLP Confusion Matrix")
plt.show()



