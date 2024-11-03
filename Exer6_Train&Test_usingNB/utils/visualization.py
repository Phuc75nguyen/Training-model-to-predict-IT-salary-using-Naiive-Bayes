import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred):
    # Vẽ biểu đồ scatter
    plt.figure(figsize=(10, 6))
    
    # Vẽ các điểm dự đoán
    plt.scatter(y_true, y_pred, alpha=0.6, label='Dự đoán', color='blue')

    # Vẽ đường chéo cho dự đoán chính xác
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Đường Dự Đoán')

    plt.title('Dự đoán so với Giá trị Thực tế')
    plt.xlabel('Giá trị Thực tế')
    plt.ylabel('Giá trị Dự đoán')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid()
    plt.legend()
    plt.show()

# Sử dụng hàm
# plot_predictions(y_test, y_pred)

