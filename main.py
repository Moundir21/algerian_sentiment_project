from dl_models.train_lstm import train_lstm_model
from dl_models.train_gru import train_gru_model

print("\n📊 Comparison:")
print("LSTM vs GRU based on accuracy and training time")

print(" \n 🔵 Training LSTM...")
#train_lstm_model()

print("\n🟢 Training GRU...")
train_gru_model()

print("\n📊 Comparison:")
print("LSTM vs GRU based on accuracy and training time")