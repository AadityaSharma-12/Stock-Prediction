import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import io, base64




device = torch.device("cpu")


# -------------------
# Model Class
# -------------------
class StockPredictor:
    def __init__(self, ticker, seq_length=30, hidden_dim=32, num_layers=2, epochs=200):
        self.ticker = ticker
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs

    class PredictionModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super().__init__()
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


    def run(self):
        # -------------------
        # Data download
        # -------------------
        df = yf.download(self.ticker, "2020-01-01")

        scaler = StandardScaler()
        df["Close_scaled"] = scaler.fit_transform(df[["Close"]])

        data = []
        for i in range(len(df) - self.seq_length):
            data.append(df.Close_scaled.values[i:i+self.seq_length])
        data = np.array(data)

        train_size = int(0.8 * len(data))
        x_train = torch.from_numpy(data[:train_size, :-1].reshape(-1, self.seq_length-1, 1)).float().to(device)
        y_train = torch.from_numpy(data[:train_size, -1].reshape(-1, 1)).float().to(device)
        x_test = torch.from_numpy(data[train_size:, :-1].reshape(-1, self.seq_length-1, 1)).float().to(device)
        y_test = torch.from_numpy(data[train_size:, -1].reshape(-1, 1)).float().to(device)

        # -------------------
        # Model + Training
        # -------------------
        model = self.PredictionModel(1, self.hidden_dim, self.num_layers, 1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for i in range(self.epochs):
            model.train()
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------------------
        # Evaluation
        # -------------------
        model.eval()
        y_test_pred = model(x_test)

        y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
        y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

        test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

        # -------------------
        # Today's price & tomorrow's prediction
        # -------------------
        today_close = df["Close"].iloc[-1].item()

        last_seq = df.Close_scaled.values[-(self.seq_length-1):].reshape(1, self.seq_length-1, 1)
        last_seq_tensor = torch.from_numpy(last_seq).float().to(device)
        tomorrow_pred = model(last_seq_tensor).detach().cpu().numpy()
        tomorrow_pred = scaler.inverse_transform(tomorrow_pred)[0][0]

        # -------------------
        # 7-day comparison table
        # -------------------
        last_7_dates = df.iloc[-len(y_test):].index[-7:].strftime('%Y-%m-%d')
        last_7_actual = y_test[-7:].flatten()
        last_7_pred = y_test_pred[-7:].flatten()

        comparison_df = pd.DataFrame({
            "Date": last_7_dates,
            "Actual Price": last_7_actual,
            "Predicted Price": last_7_pred
        }).reset_index(drop=True)
        comparison_df = comparison_df.round(2)

        # -------------------
        # Chart
        # -------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.iloc[-len(y_test):].index, y_test, label="Actual Price", color="blue")
        ax.plot(df.iloc[-len(y_test):].index, y_test_pred, label="Predicted Price", color="green")
        ax.legend()
        ax.set_title(f"{self.ticker} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        return today_close, tomorrow_pred, comparison_df, plot_url



def run_prediction(ticker):
    predictor = StockPredictor(ticker)
    return predictor.run()


if __name__ == "__main__":
    # Default ticker only for console runs
    ticker = "AAPL"
    today, tomorrow, table, _ = run_prediction(ticker)
    print(f"\nToday's closing price: {today:.2f}")
    print(f"Predicted price for tomorrow: {tomorrow:.2f}")
    print("\nLast 7 Days Comparison:\n")
    print(table.to_string(index=False))
