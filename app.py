import streamlit as dt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import io

# 1. SET PAGE CONFIG (Lệnh Streamlit đầu tiên)
dt.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# 2. IMPORT & CÁC HÀM CACHE DÙNG CHUNG
@dt.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo khả năng hash của cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        dt.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# 3. SIDEBAR (THÀNH PHẦN 1: VÙNG CẤU HÌNH)
with dt.sidebar:
    dt.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải file dữ liệu huấn luyện mẫu
    uploaded_file = dt.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn file dữ liệu chứa các tính năng giao dịch và cột nhãn 'default'."
    )
    
    dt.markdown("---")
    dt.subheader("Tham số mô hình AI")
    dt.caption("Mô hình: Random Forest Classifier (Theo Notebook)")
    
    # Các siêu tham số trích xuất từ cấu trúc mô hình tối ưu mặc định
    n_estimators = dt.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    max_depth = dt.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=1, 
        max_value=50, 
        value=15, 
        step=1,
        help="Độ sâu tối đa của mỗi cây quyết định."
    )
    
    random_state = dt.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        value=42, 
        step=1,
        help="Giá trị seed để cố định kết quả phân tách dữ liệu và huấn luyện."
    )
    
    test_size = dt.slider(
        "Tỷ lệ dữ liệu kiểm thử (test_size)", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.2, 
        step=0.05,
        help="Tỷ lệ chia tập dữ liệu thành tập kiểm thử (Test Set)."
    )
    
    dt.divider()
    # Nút hành động huấn luyện duy nhất
    btn_train = dt.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu xử lý dữ liệu và huấn luyện mô hình với tham số đã chọn."
    )

# 4. HEADER (THÀNH PHẦN 2: VÙNG ĐỊNH HƯỚNG)
dt.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận & Rủi ro tín dụng")
dt.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch, đánh giá mô hình học máy Random Forest và dự báo trực tuyến rủi ro gian lận/nợ xấu (Cột mục tiêu: 'default').")

# Kiểm tra trạng thái dữ liệu đầu vào hành vi người dùng
if uploaded_file is None:
    dt.info("💡 Vui lòng tải lên tệp dữ liệu huấn luyện (.csv hoặc .xlsx) ở thanh cấu hình bên trái để bắt đầu.")
    dt.stop()

# Đọc dữ liệu qua hàm cache chung khi có file
file_bytes = uploaded_file.read()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    dt.error("Không thể xử lý định dạng tệp này. Vui lòng kiểm tra lại cấu trúc file.")
    dt.stop()

# Kiểm tra sự tồn tại của biến mục tiêu 'default'
if 'default' not in df_raw.columns:
    dt.error("❌ Tệp dữ liệu thiếu cột mục tiêu 'default'. Vui lòng chọn đúng tệp cấu trúc.")
    dt.stop()

dt.caption(f"📁 Đang dùng tệp: `{uploaded_file.name}` | Quy mô dữ liệu: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột.")
dt.divider()

# 5. KHỐI HUẤN LUYỆN (Chạy khi bấm nút, lưu kết quả vào session_state)
# Tập biến đầu vào chính xác theo cấu trúc tệp dữ liệu được phân tích (X_1 đến X_14)
feature_cols = [f"X_{i}" for i in range(1, 15)]
missing_features = [col for col in feature_cols if col not in df_raw.columns]

if missing_features:
    dt.error(f"❌ Thiếu các cột tính năng bắt buộc sau trong dữ liệu đầu vào: {missing_features}")
    dt.stop()

if btn_train:
    with dt.spinner("🔄 Đang xử lý dữ liệu và huấn luyện mô hình..."):
        # Phân tách tập dữ liệu
        X = df_raw[feature_cols].copy()
        y = df_raw['default'].copy()
        
        # Xử lý giá trị khuyết thiếu cục bộ nếu có (Điền bằng median)
        X = X.fillna(X.median())
        y = y.fillna(y.mode()[0])
        
        # Chia tập Train - Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Tiền xử lý: Chuẩn hóa dữ liệu StandardScaler giống Pipeline thực tế
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Huấn luyện mô hình Random Forest Classifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Đánh giá kết quả trên tập kiểm thử
        y_pred = model.predict(X_test_scaled)
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        except:
            y_prob = None
            
        # Lưu 3 thành phần bắt buộc vào session_state
        dt.session_state['trained_model'] = model
        dt.session_state['data_scaler'] = scaler
        dt.session_state['metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else None,
            'feature_importances': model.feature_importances_.tolist()
        }
        dt.success("🎉 Huấn luyện mô hình thành công! Hãy xem kết quả chi tiết ở các Tab bên dưới.")

# 6. KHỞI TẠO TABS CHỨA CÁC THÀNH PHẦN NỘI DUNG
tab_overview, tab_viz, tab_eval, tab_predict = dt.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
with tab_overview:
    dt.subheader("🗂️ Phân tích thống kê và Cấu trúc tệp dữ liệu")
    
    # Chỉ số quy mô
    col_m1, col_m2, col_m3 = dt.columns(3)
    file_size_mb = len(file_bytes) / (1024 * 1024)
    col_m1.metric("Số hàng dữ liệu", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng biến", f"{df_raw.shape[1]:,}")
    col_m3.metric("Dung lượng tệp tải lên", f"{file_size_mb:.2f} MB")
    
    # Hiển thị dữ liệu thô đầu vào
    dt.markdown("### 🗄️ Xem trước dữ liệu thô (5 dòng đầu tiên)")
    dt.dataframe(df_raw.head(5), use_container_width=True)
    
    # Thống kê mô tả tập dữ liệu mô hình
    dt.markdown("### 📉 Chỉ số mô tả các biến đặc trưng đưa vào mô hình")
    cols_to_describe = feature_cols + ['default']
    dt.dataframe(df_raw[cols_to_describe].describe(), use_container_width=True)

# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
with tab_viz:
    dt.subheader("📊 Trực quan phân phối tính năng & Biến mục tiêu")
    
    # 1. Vẽ phân phối biến mục tiêu
    fig_target = px.histogram(
        df_raw, x='default', 
        color='default',
        title="Phân phối của biến mục tiêu (0: Bình thường, 1: Gian lận/Rủi ro)",
        labels={'default': 'Trạng thái default'},
        color_discrete_map={0: '#1f77b4', 1: '#d62728'}
    )
    fig_target.update_layout(height=350)
    dt.plotly_chart(fig_target, use_container_width=True)
    
    dt.markdown("### 📊 Biểu đồ phân phối chi tiết các biến đầu vào")
    # Widget cho phép lựa chọn linh động nếu số biến lớn
    selected_features = dt.multiselect(
        "Chọn các biến đầu vào muốn hiển thị (Mặc định hiển thị 4 biến đầu tiên)",
        options=feature_cols,
        default=feature_cols[:4]
    )
    
    if selected_features:
        # Bố trí biểu đồ dạng lưới cân đối
        num_cols_layout = 2
        cols = dt.columns(num_cols_layout)
        for idx, col_name in enumerate(selected_features):
            current_col = cols[idx % num_cols_layout]
            with current_col:
                # Kiểm tra kiểu số liên tục để vẽ Histogram phân phối kèm nhãn màu
                fig_feat = px.histogram(
                    df_raw, x=col_name, color='default',
                    marginal="box",
                    title=f"Phân phối biến {col_name} theo nhóm mục tiêu",
                    color_discrete_map={0: '#2ca02c', 1: '#ff7f0e'},
                    barmode='overlay'
                )
                fig_feat.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                dt.plotly_chart(fig_feat, use_container_width=True)
    else:
        dt.warning("Vui lòng lựa chọn ít nhất một biến đầu vào để trực quan hóa.")

# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
with tab_eval:
    dt.subheader("🎯 Đánh giá hiệu năng mô hình phân loại")
    
    # Điều phối kiểm tra trạng thái huấn luyện
    if 'trained_model' not in dt.session_state:
        dt.info("💡 Vui lòng thiết lập cấu hình tham số và bấm nút **[🚀 Huấn luyện mô hình]** ở thanh Sidebar bên trái để hiển thị kết quả kiểm định.")
    else:
        metrics = dt.session_state['metrics']
        
        # Trình bày chỉ tiêu vô hướng qua Metric Cards
        col_a, col_b, col_c, col_d = dt.columns(4)
        col_a.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        col_b.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.4f}")
        col_c.metric("Độ nhạy (Recall)", f"{metrics['recall']:.4f}")
        col_d.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        dt.markdown("---")
        
        col_left, col_right = dt.columns(2)
        
        with col_left:
            dt.markdown("### 🧩 Ma trận nhầm lẫn (Confusion Matrix)")
            y_test_arr = np.array(metrics['y_test'])
            y_pred_arr = np.array(metrics['y_pred'])
            cm = confusion_matrix(y_test_arr, y_pred_arr)
            
            # Trực quan Confusion Matrix bằng Plotly Heatmap
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['Bình thường (0)', 'Gian lận/Rủi ro (1)'],
                y=['Bình thường (0)', 'Gian lận/Rủi ro (1)'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=350)
            dt.plotly_chart(fig_cm, use_container_width=True)
            
        with col_right:
            dt.markdown("### 🌲 Độ quan trọng của các biến (Feature Importance)")
            importance_df = pd.DataFrame({
                'Tính năng': feature_cols,
                'Độ quan trọng': metrics['feature_importances']
            }).sort_values(by='Độ quan trọng', ascending=True)
            
            fig_imp = px.bar(
                importance_df,
                x='Độ quan trọng',
                y='Tính năng',
                orientation='h',
                title='Mức độ đóng góp của từng biến vào mô hình',
                color='Độ quan trọng',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(height=350, showlegend=False)
            dt.plotly_chart(fig_imp, use_container_width=True)

# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
with tab_predict:
    dt.subheader("🔮 Dự báo rủi ro gian lận giao dịch cho dữ liệu mới")
    
    if 'trained_model' not in dt.session_state:
        dt.info("💡 Vui lòng hoàn thành bước huấn luyện mô hình để kích hoạt tính năng dự báo.")
    else:
        model = dt.session_state['trained_model']
        scaler = dt.session_state['data_scaler']
        
        mode = dt.radio(
            "Chọn chế độ nhập dữ liệu:",
            ["👉 Nhập trực tiếp từng bản ghi", "📁 Tải lên tệp danh sách cần dự báo hàng loạt"],
            horizontal=True
        )
        
        # Lấy giá trị thống kê nền tảng để điền mặc định cho form nhập liệu ổn định
        median_values = df_raw[feature_cols].median().to_dict()
        min_values = df_raw[feature_cols].min().to_dict()
        max_values = df_raw[feature_cols].max().to_dict()
        
        if mode == "👉 Nhập trực tiếp từng bản ghi":
            dt.markdown("### 📝 Điền thông số các biến đặc trưng giao dịch")
            
            with dt.form("prediction_form"):
                # Gom các biến vào các cột phân bố gọn gàng trong Form giao diện
                p_cols = dt.columns(3)
                user_inputs = {}
                
                for idx, col_name in enumerate(feature_cols):
                    current_p_col = p_cols[idx % 3]
                    with current_p_col:
                        # Tất cả các biến trong tập dữ liệu đều thuộc dạng số liên tục/rời rạc float64
                        user_inputs[col_name] = dt.number_input(
                            f"Nhập giá trị {col_name}",
                            min_value=float(min_values[col_name] * 5), # Mở rộng biên an toàn cho người dùng nhập liệu
                            max_value=float(max_values[col_name] * 5),
                            value=float(median_values[col_name]),
                            format="%.6f",
                            help=f"Giá trị mặc định dựa trên trung vị dữ liệu mẫu: {median_values[col_name]:.4f}"
                        )
                
                submit_predict = dt.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary")
                
            if submit_predict:
                # Chuyển đổi dữ liệu nhập thành DataFrame
                input_df = pd.DataFrame([user_inputs])
                
                # Áp dụng chính xác bộ tiền xử lý chuẩn hóa đã học lúc train
                input_scaled = scaler.transform(input_df)
                
                # Thực hiện dự báo mô hình
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0][1]
                
                # Hiển thị kết quả trực quan sinh động
                dt.markdown("---")
                dt.markdown("### 📋 Kết quả đánh giá từ hệ thống:")
                
                c1, c2 = dt.columns(2)
                if prediction == 1:
                    c1.error("🚨 CẢNH BÁO: Giao dịch có dấu hiệu nguy cơ GIAN LẬN / RỦI RO CAO!")
                    c2.metric("Xác suất rủi ro", f"{proba * 100:.2f} %", delta="Nguy hiểm", delta_color="inverse")
                else:
                    c1.success("✅ AN TOÀN: Giao dịch được đánh giá là Bình thường.")
                    c2.metric("Xác suất rủi ro", f"{proba * 100:.2f} %", delta="An tâm", delta_color="normal")
                    
        elif mode == "📁 Tải lên tệp danh sách cần dự báo hàng loạt":
            dt.markdown("### 📂 Tải lên file chứa cấu trúc danh sách biến đầu vào")
            dt.caption("Lưu ý: File tải lên bắt buộc phải chứa đầy đủ các cột từ `X_1` tới `X_14`.")
            
            predict_file = dt.file_uploader(
                "Chọn file dữ liệu cần kiểm định hàng loạt (.csv, .xlsx)", 
                type=["csv", "xlsx"],
                key="bulk_predict_uploader"
            )
            
            if predict_file is not None:
                predict_bytes = predict_file.read()
                df_predict_raw = load_data(predict_bytes, predict_file.name)
                
                if df_predict_raw is not None:
                    # Kiểm tra sự trùng khớp của cấu trúc cột dữ liệu
                    missing_p_features = [col for col in feature_cols if col not in df_predict_raw.columns]
                    
                    if missing_p_features:
                        dt.error(f"❌ File tải lên không hợp lệ. Thiếu các cột tính năng bắt buộc sau: {missing_p_features}")
                    else:
                        # Tạo bản sao tính toán độc lập
                        df_features = df_predict_raw[feature_cols].copy().fillna(df_raw[feature_cols].median())
                        
                        # Chuẩn hóa dữ liệu với cùng bộ scaler
                        df_features_scaled = scaler.transform(df_features)
                        
                        # Dự báo hàng loạt
                        bulk_preds = model.predict(df_features_scaled)
                        bulk_probs = model.predict_proba(df_features_scaled)[:, 1]
                        
                        # Nhúng kết quả trực tiếp vào bảng phân phối kết xuất
                        df_output = df_predict_raw.copy()
                        df_output['Kết quả dự báo (Mô hình)'] = bulk_preds
                        df_output['Xác suất rủi ro gian lận'] = bulk_probs
                        
                        dt.markdown("### 📊 Danh sách kết quả dự báo tổng hợp")
                        dt.dataframe(df_output, use_container_width=True)
                        
                        # Chuyển đổi dữ liệu để tải về dưới dạng CSV
                        csv_buffer = io.StringIO()
                        df_output.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        dt.download_button(
                            label="📥 Tải xuống kết quả dự báo dạng (.CSV)",
                            data=csv_data,
                            file_name="Ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
