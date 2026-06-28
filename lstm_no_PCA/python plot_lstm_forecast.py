import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Đọc dữ liệu LSTM
file_path = r"C:\Users\ADMIN\Desktop\CAPSTONE PROJECT\outputs\lstm_vnindex_sweep\predictions_lookback_45_batch_16.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])

# Vẽ biểu đồ
plt.figure(figsize=(14, 6))

# Vẽ đường thực tế - màu xanh, nét liền
plt.plot(df['Date'], df['Actual_VNINDEX'], 
         label="Actual VNINDEX", 
         linewidth=2, 
         color='blue')

# Vẽ đường dự báo LSTM - màu đỏ, nét đứt (để phân biệt)
plt.plot(df['Date'], df['Predicted_VNINDEX'], 
         label="LSTM Forecast (LB=45, BS=16)", 
         linewidth=2, 
         color='red', 
         linestyle='--')

# Tiêu đề và nhãn
plt.title("VNINDEX Forecast on Test Set (LSTM + PCA, LB=45, BS=16)")
plt.xlabel("Date")
plt.ylabel("VNINDEX")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)

# Định dạng trục x
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

# Điều chỉnh layout
plt.tight_layout()

# Lưu ảnh
output_dir = Path(r"C:\Users\ADMIN\Desktop\CAPSTONE PROJECT\outputs\lstm_vnindex_sweep\plots")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "lstm_forecast_plot.png", dpi=300, bbox_inches='tight')
print(f"✅ Đã lưu biểu đồ LSTM tại: {output_dir / 'lstm_forecast_plot.png'}")

plt.show()