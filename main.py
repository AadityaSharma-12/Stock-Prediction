import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device("cpu")
#USE IF SYSTEM HAS A GPU - device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------
# Data download
# -------------------
ticker = 'AAPL' #Use ticker of any company "Apple(AAPL) in this case"
df = yf.download(ticker, '2020-01-01')

# Plot raw prices
df.Close.plot(figsize=(12, 8), title=f"{ticker} Stock Price")

# -------------------
# Preprocessing
# -------------------
scaler = StandardScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

seq_length = 30
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close_scaled.values[i:i+seq_length])

data = np.array(data)

train_size = int(0.8 * len(data))

x_train = torch.from_numpy(data[:train_size, :-1].reshape(-1, seq_length-1, 1)).float().to(device)
y_train = torch.from_numpy(data[:train_size, -1].reshape(-1, 1)).float().to(device)
x_test = torch.from_numpy(data[train_size:, :-1].reshape(-1, seq_length-1, 1)).float().to(device)
y_test = torch.from_numpy(data[train_size:, -1].reshape(-1, 1)).float().to(device)

# -------------------
# Model
# -------------------
class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------
# Training
# -------------------
num_epochs = 200
for i in range(num_epochs):
    model.train()
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)
    if i % 25 == 0:
        print(f"Epoch {i}, Loss: {loss.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -------------------
# Evaluation
# -------------------
model.eval()
y_test_pred = model(x_test)

# Inverse transform
y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

#print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

# -------------------
# Today's Closing Price & Predict Tomorrow
# -------------------
today_close = df['Close'].iloc[-1].item()
print(f"\nToday's closing price: {today_close:.2f}")

last_seq = df.Close_scaled.values[-(seq_length-1):].reshape(1, seq_length-1, 1)
last_seq_tensor = torch.from_numpy(last_seq).float().to(device)
tomorrow_pred = model(last_seq_tensor).detach().cpu().numpy()
tomorrow_pred = scaler.inverse_transform(tomorrow_pred)[0][0]

print(f"Predicted price for tomorrow: {tomorrow_pred:.2f}")

# -------------------
# Plot Results
# -------------------
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(5, 1)

# Price vs Prediction
ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(df.iloc[-len(y_test):].index, y_test, color='blue', label='Actual price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color='green', label='Predicted price')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Price')

# Error Plot
ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), 'r', label='Prediction Error')
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')

# -------------------
# Table of last 7 days
# -------------------
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

last_7_dates = df.iloc[-len(y_test):].index[-7:].strftime('%Y-%m-%d')
last_7_actual = y_test[-7:].flatten()
last_7_pred = y_test_pred[-7:].flatten()

comparison_df = pd.DataFrame({
    "Date": last_7_dates,
    "Actual Price": last_7_actual,
    "Predicted Price": last_7_pred
}).reset_index(drop=True)

print("\nLast 7 Days Comparison:\n")
print(comparison_df.to_string(index=False))


plt.tight_layout()
plt.show()
