

import pandas as pd

# các hàm để chuẩn bị data IT salary

# hàm gộp các file IT salary thành 1 file dữ liệu

# data_preparation.py
import pandas as pd
from utils.data_cleaning import clean_2018_data, clean_2019_data, clean_2020_data

def load_concat_data():
    # Đường dẫn tuyệt đối cho các file dữ liệu
    files = [
        "D:/COMPUTER SCIENCE PTITHCM/PYTHON4AI/Data Science/Exer6_Train&Test_usingNB/data/IT_Salary_Survey_EU_2018.csv",
        "D:/COMPUTER SCIENCE PTITHCM/PYTHON4AI/Data Science/Exer6_Train&Test_usingNB/data/IT_Salary_Survey_EU_2019.csv",
        "D:/COMPUTER SCIENCE PTITHCM/PYTHON4AI/Data Science/Exer6_Train&Test_usingNB/data/IT_Salary_Survey_EU_2020.csv"
    ]
    
    # Đọc dữ liệu từ các file và làm sạch từng phần
    df_2018 = clean_2018_data(pd.read_csv(files[0]))
    df_2019 = clean_2019_data(pd.read_csv(files[1]))
    df_2020 = clean_2020_data(pd.read_csv(files[2]))
    
    # Gộp các DataFrame đã làm sạch
    combined_data = pd.concat([df_2018, df_2019, df_2020], ignore_index=True)
    return combined_data

def handle_missing_data(data):
    # Convert 'Timestamp' to datetime và thêm cột 'year' nếu có
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce', dayfirst=True)
        data['year'] = data['Timestamp'].dt.year
        data = data[data['year'].isin([2018, 2019, 2020])].copy()

    # Điền missing values cho các cột số bằng trung bình
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].apply(lambda x: x.fillna(x.mean()))

    # Điền missing values cho các cột phân loại với mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if not data[col].mode().empty:
            mode_value = data[col].mode()[0]
            data.loc[:, col] = data[col].fillna(mode_value)
        else:
            data.loc[:, col] = data[col].fillna('')  # Điền chuỗi rỗng nếu không có mode

    return data

# Hàm hoàn chỉnh để chuẩn bị dữ liệu tổng hợp và xử lý missing values
def prepare_data():
    data = load_concat_data()
    data = handle_missing_data(data)
    return data
