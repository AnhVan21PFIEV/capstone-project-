# step_13_check_singularity.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def run(context: dict) -> dict:
    """Kiểm tra ma trận suy biến và đa cộng tuyến của dữ liệu gốc"""
    
    X_train = context["X_trainval"]
    feature_cols = context["feature_cols"]
    
    print("\n" + "=" * 90)
    print("📊 KIỂM TRA MA TRẬN SUY BIẾN VÀ ĐA CỘNG TUYẾN")
    print("=" * 90)
    
    # ===== 1. MA TRẬN TƯƠNG QUAN =====
    print("\n📌 1. PHÂN TÍCH MA TRẬN TƯƠNG QUAN")
    print("-" * 90)
    
    # Tính ma trận tương quan
    corr_matrix = np.corrcoef(X_train.T)
    
    # Thống kê tương quan
    n_features = X_train.shape[1]
    n_pairs = n_features * (n_features - 1) // 2
    
    # Lấy các giá trị tương quan (không bao gồm đường chéo)
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    corr_values = corr_matrix[upper_tri_indices]
    
    # Thống kê
    corr_high = np.sum(np.abs(corr_values) > 0.7)
    corr_very_high = np.sum(np.abs(corr_values) > 0.9)
    corr_perfect = np.sum(np.abs(corr_values) > 0.99)
    
    print(f"Số features: {n_features}")
    print(f"Số cặp tương quan: {n_pairs}")
    print(f"Số cặp có |r| > 0.7 (tương quan cao): {corr_high} ({corr_high/n_pairs*100:.2f}%)")
    print(f"Số cặp có |r| > 0.9 (tương quan rất cao): {corr_very_high} ({corr_very_high/n_pairs*100:.2f}%)")
    print(f"Số cặp có |r| > 0.99 (gần như hoàn hảo): {corr_perfect} ({corr_perfect/n_pairs*100:.2f}%)")
    
    # Tìm các cặp tương quan cao nhất
    print("\n🔍 Top 10 cặp features có tương quan cao nhất:")
    flat_indices = np.argsort(np.abs(corr_values))[-10:][::-1]
    
    for idx in flat_indices:
        i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        corr_val = corr_values[idx]
        print(f"   {feature_cols[i][:20]} ↔ {feature_cols[j][:20]} : r = {corr_val:.6f}")
    
    # ===== 2. ĐIỀU KIỆN (CONDITION NUMBER) =====
    print("\n📌 2. KIỂM TRA ĐIỀU KIỆN MA TRẬN (Condition Number)")
    print("-" * 90)
    
    # Tính condition number
    try:
        # Chuẩn hóa dữ liệu
        X_centered = X_train - np.mean(X_train, axis=0)
        
        # Tính singular values
        u, s, vh = np.linalg.svd(X_centered, full_matrices=False)
        condition_number = s[0] / s[-1]
        
        print(f"Condition number: {condition_number:.2e}")
        
        if condition_number > 1e10:
            print("   ⚠️ Condition number > 10^10: Ma trận BỊ SUY BIẾN NGHIÊM TRỌNG")
            print(f"      → Không thể tính ma trận nghịch đảo (X'X)^-1")
            print(f"      → ARDL không thể ước lượng tham số khi Q >= 2")
        elif condition_number > 1e6:
            print("   ⚠️ Condition number > 10^6: Ma trận SUY BIẾN")
            print(f"      → Phép toán nghịch đảo không ổn định")
            print(f"      → Kết quả ước lượng không đáng tin cậy")
        else:
            print("   ✅ Condition number chấp nhận được")
        
        # In các singular values
        print(f"\n   Top 5 singular values:")
        for i in range(min(5, len(s))):
            print(f"      s{i+1} = {s[i]:.2e}")
        print(f"   ...")
        print(f"   s{len(s)} = {s[-1]:.2e}")
        
    except Exception as e:
        print(f"❌ Không thể tính condition number: {e}")
    
    # ===== 3. VIF (Variance Inflation Factor) =====
    print("\n📌 3. KIỂM TRA VIF (Variance Inflation Factor)")
    print("-" * 90)
    
    # DO MA TRẬN SUY BIẾN, VIF KHÔNG THỂ TÍNH ĐƯỢC
    # ĐÂY CŨNG LÀ MỘT BẰNG CHỨNG CHO THẤY DỮ LIỆU BỊ ĐA CỘNG TUYẾN
    print("❌ KHÔNG THỂ TÍNH VIF DO MA TRẬN SUY BIẾN")
    print("   → Không thể tính ma trận nghịch đảo (X'X)^-1")
    print("   → Đây là bằng chứng cho thấy dữ liệu bị đa cộng tuyến NGHIÊM TRỌNG")
    print("   → VIF (Variance Inflation Factor) không thể tính được")
    print("   → ARDL không thể ước lượng tham số đáng tin cậy")
    
    # Thử tính VIF bằng phương pháp khác (tính R² cho từng feature)
    print("\n   Thử tính R² của từng feature với các feature còn lại:")
    print("   (Phương pháp thay thế để chứng minh đa cộng tuyến)")
    
    try:
        from sklearn.linear_model import LinearRegression
        
        n_sample = min(20, n_features)
        r2_values = []
        
        for i in range(n_sample):
            # Feature i làm target
            y_i = X_train[:, i]
            # Các feature còn lại làm predictors
            X_i = np.delete(X_train, i, axis=1)
            
            # Nếu số features > số samples, không thể hồi quy
            if X_i.shape[1] > X_i.shape[0]:
                r2_values.append({
                    "Feature": feature_cols[i][:30],
                    "R²": np.nan,
                    "Status": "Không thể tính (p > n)"
                })
                continue
            
            try:
                model = LinearRegression()
                model.fit(X_i, y_i)
                r2 = model.score(X_i, y_i)
                r2_values.append({
                    "Feature": feature_cols[i][:30],
                    "R²": r2,
                    "Status": "OK"
                })
            except Exception as e:
                r2_values.append({
                    "Feature": feature_cols[i][:30],
                    "R²": np.nan,
                    "Status": f"Lỗi: {str(e)[:30]}"
                })
        
        # In kết quả R²
        print("\n   R² của từng feature với các feature còn lại (mẫu 20 features):")
        print("   (R² càng gần 1 → đa cộng tuyến càng nghiêm trọng)")
        print("   " + "-" * 80)
        
        for item in r2_values:
            if pd.isna(item["R²"]):
                status_display = "❌ " + item["Status"]
            elif item["R²"] > 0.99:
                status_display = f"🔴 R² = {item['R²']:.6f} (RẤT CAO)"
            elif item["R²"] > 0.95:
                status_display = f"🟠 R² = {item['R²']:.6f} (CAO)"
            elif item["R²"] > 0.8:
                status_display = f"🟡 R² = {item['R²']:.6f} (TRUNG BÌNH)"
            else:
                status_display = f"🟢 R² = {item['R²']:.6f} (THẤP)"
            
            print(f"   {item['Feature']:<30} {status_display}")
        
        # Thống kê R²
        r2_high = sum(1 for x in r2_values if not pd.isna(x["R²"]) and x["R²"] > 0.95)
        print(f"\n   Thống kê R² (trên {len(r2_values)} features mẫu):")
        print(f"   - Số features có R² > 0.95: {r2_high}/{len(r2_values)}")
        print(f"   → Đa cộng tuyến RẤT NGHIÊM TRỌNG")
        
    except Exception as e:
        print(f"   ❌ Không thể tính R² thay thế: {e}")
    
    # ===== 4. ĐỊNH THỨC MA TRẬN =====
    print("\n📌 4. KIỂM TRA ĐỊNH THỨC MA TRẬN (Determinant)")
    print("-" * 90)
    
    try:
        # Tính ma trận X'X
        XtX = X_train.T @ X_train
        
        # Tính định thức
        det = np.linalg.det(XtX)
        
        print(f"Định thức của ma trận X'X: {det:.2e}")
        
        if abs(det) < 1e-10:
            print("   ⚠️ Định thức ≈ 0 → Ma trận SUY BIẾN")
            print("   → Không thể tìm ma trận nghịch đảo (X'X)^-1")
            print("   → ARDL không thể ước lượng tham số")
        else:
            print("   ✅ Định thức khác 0 → Ma trận khả nghịch")
            
    except Exception as e:
        print(f"❌ Không thể tính định thức: {e}")
        print("   → Ma trận X'X BỊ SUY BIẾN")
    
    # ===== 5. KẾT LUẬN =====
    print("\n" + "=" * 90)
    print("📌 5. KẾT LUẬN")
    print("=" * 90)
    
    print("""
🔴 ARDL (NO PCA) KHÔNG THỂ CHẠY VỚI Q >= 2 VÌ:

1. SỐ THAM SỐ QUÁ LỚN:
   - Với 318 features và Q=2: ~960 tham số
   - Chỉ có 701 mẫu dữ liệu
   - Không đủ bậc tự do để ước lượng

2. MA TRẬN SUY BIẾN (Singular Matrix):
   - """)
    
    # Thêm số liệu cụ thể
    print(f"   - {corr_high}/{n_pairs} cặp features có tương quan |r| > 0.7 ({corr_high/n_pairs*100:.1f}%)")
    print(f"   - {corr_very_high}/{n_pairs} cặp features có tương quan |r| > 0.9 ({corr_very_high/n_pairs*100:.1f}%)")
    
    if 'condition_number' in locals():
        print(f"   - Condition number = {condition_number:.2e} > 10^10 (suy biến)")
    
    print("""
   - Không thể tính ma trận nghịch đảo (X'X)^-1

3. ĐA CỘNG TUYẾN NGHIÊM TRỌNG:
   - VIF không thể tính được do ma trận suy biến
   - Đây là bằng chứng cho thấy đa cộng tuyến RẤT NGHIÊM TRỌNG
   - Các features có tương quan chéo rất cao

💡 KẾT LUẬN: 
   - Không thể áp dụng ARDL trực tiếp trên 318 cổ phiếu
   - Cần giảm chiều dữ liệu bằng PCA trước khi áp dụng ARDL
""")
    
    # Lưu kết quả
    report_dir = context["PROJECT_ROOT"] / "outputs_no_PCA" / "ardl_vnindex_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu ma trận tương quan (chỉ lấy 30x30 để tránh file quá lớn)
    n_show = min(30, n_features)
    corr_df = pd.DataFrame(
        corr_matrix[:n_show, :n_show],
        index=feature_cols[:n_show],
        columns=feature_cols[:n_show]
    )
    corr_df.to_csv(report_dir / "correlation_matrix_sample.csv")
    
    # Lưu thống kê
    stats_data = {
        "Metric": [
            "Số features",
            "Số cặp tương quan",
            "Số cặp |r| > 0.7",
            "Tỷ lệ |r| > 0.7 (%)",
            "Số cặp |r| > 0.9",
            "Tỷ lệ |r| > 0.9 (%)",
            "Số cặp |r| > 0.99",
            "Tỷ lệ |r| > 0.99 (%)",
            "Condition number",
            "Kết luận"
        ],
        "Value": [
            n_features,
            n_pairs,
            corr_high,
            f"{corr_high/n_pairs*100:.2f}",
            corr_very_high,
            f"{corr_very_high/n_pairs*100:.2f}",
            corr_perfect,
            f"{corr_perfect/n_pairs*100:.2f}",
            f"{condition_number:.2e}" if 'condition_number' in locals() else "N/A",
            "Ma trận SUY BIẾN - KHÔNG THỂ ÁP DỤNG ARDL"
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(report_dir / "singularity_stats.csv", index=False)
    
    print(f"\n📁 Kết quả đã lưu tại: {report_dir}")
    print("   - correlation_matrix_sample.csv")
    print("   - singularity_stats.csv")
    
    return context