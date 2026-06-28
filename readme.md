# Application of Dimensionality Reduction and Deep Learning Techniques in Forecasting the VN-INDEX

Dự án sử dụng mô hình ARDL (Autoregressive Distributed Lag model ) và LSTM (Long Short-Term Memory) kết hợp với PCA (Principal Component Analysis) để dự báo chỉ số VN-Index.

## Yêu cầu hệ thống
+ Python : 3.11+

## Cài đặt 
### 1. Clone repository (hoặc tải source code)
```bash
git clone <repository_url>
cd <project_folder>
```
### 2. Tạo thư mục data ở project_folder
```bash
cd data
mkdir raw # bỏ Data_VNINDEX.csv vào đây (398 mã)
mkdir processed

```
### 2. Tải thư viện cần thiết 

```bash
pip install -r requirements_ardl.txt
```
### 3. Tiền xử lý + PCA model 
```bash
cd src
python src/run_all.py --config config/config.yaml
```
### 5. ARDL  
```bash
cd ardl 
python run_all_ardl.py  
```
### 5. LSTM 
```bash
cd lstm 
python run_all_lstm.py  
```

  - `README.md`          # Tài liệu hướng dẫn sử dụng
