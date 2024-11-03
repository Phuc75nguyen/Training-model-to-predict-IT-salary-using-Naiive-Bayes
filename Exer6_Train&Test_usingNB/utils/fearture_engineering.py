# feature_engineering.py
import pandas as pd

def remove_low_correlation_features(data, target_column, threshold=0.1):
    """
    Loại bỏ các cột có độ tương quan thấp với cột mục tiêu, giữ lại `salary_category` nếu có.
    
    Parameters:
    - data: DataFrame chứa dữ liệu.
    - target_column: Tên cột mục tiêu để tính toán độ tương quan.
    - threshold: Ngưỡng tương quan (mặc định là 0.1). Các cột có độ tương quan thấp hơn ngưỡng này sẽ bị loại bỏ.
    
    Returns:
    - DataFrame chỉ bao gồm các cột có độ tương quan cao hơn threshold với cột mục tiêu.
    """
    # Lọc các cột số cho ma trận tương quan
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    
    # Chỉ giữ lại các cột có tương quan lớn hơn threshold với cột target
    columns_to_keep = correlation_matrix[abs(correlation_matrix[target_column]) > threshold].index
    
    # Đảm bảo giữ lại cột 'salary_category' nếu tồn tại
    columns_to_keep = columns_to_keep.union(['salary_category']) if 'salary_category' in data.columns else columns_to_keep
    
    return data[columns_to_keep]


def convert_categorical_to_numeric(data):
    """
    Chuyển đổi các biến phân loại thành các biến giả (one-hot encoding).
    
    Parameters:
    - data: DataFrame chứa dữ liệu.

    Returns:
    - DataFrame sau khi áp dụng one-hot encoding cho các biến phân loại.
    """
    return pd.get_dummies(data, drop_first=True)
