import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Sample data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model
model = nn.Linear(in_features=1, out_features=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Print learned parameters
print("Weight:", model.weight.item(), "Bias:", model.bias.item())

# Plotting
import matplotlib.pyplot as plt

predicted = model(x).detach().numpy()
plt.scatter(x.numpy(), y.numpy(), label="Original data")
plt.plot(x.numpy(), predicted, label="Fitted line", color='red')
plt.legend()
plt.show()
