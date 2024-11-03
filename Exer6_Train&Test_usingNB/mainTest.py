from sklearn.model_selection import train_test_split
from utils.data_preparation import load_concat_data, handle_missing_data
from utils.fearture_engineering import convert_categorical_to_numeric
from utils.model_training import train_naive_bayes, evaluate_model
from utils.visualization import plot_predictions

# Define categorize_salary function to use NB for trainig model :((


def categorize_salary(salary):
    if salary < 50000:
        return 'Low'
    elif 50000 <= salary <= 100000:
        return 'Medium'
    else:
        return 'High'


# Load and prepare data
data = load_concat_data()  # Load and concatenate data from multiple years
data = handle_missing_data(data)  # Handle missing values

# Add 'salary_category' column with categorical values
data['salary_category'] = data['salary'].apply(categorize_salary)

# Remove 'salary' column to avoid confusion with classification
data = data.drop('salary', axis=1)

# Now you can print the value counts of 'salary_category'
print(data['salary_category'].value_counts())

# Ensure 'salary_category' remains as a single categorical column
print("Columns after processing:", data.columns)

# Remove DateTime columns if present or convert them to numeric (timestamp)
date_columns = data.select_dtypes(include=['datetime64[ns]']).columns
# Alternatively, convert them to timestamps if needed
data = data.drop(date_columns, axis=1)


# data = remove_low_correlation_features(data, target_column='salary_category')

# Separate features and target
X = data.drop('salary_category', axis=1)
y = data['salary_category']  # This should now be a 1D array

# Convert other categorical columns in X to numeric format (exclude target column 'salary_category')
X = convert_categorical_to_numeric(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model and evaluate
model = train_naive_bayes(X_train, y_train)
accuracy, classification_report, confusion_matrix = evaluate_model(
    model, X_test, y_test)

# Display evaluation results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report)
print("\nConfusion Matrix:\n", confusion_matrix)

# Visualizations

plot_predictions(y_test, model.predict(X_test))
# plot_correlation_heatmap(data, target_column='salary_category')

# Ghi chú: Không vẽ plot_residuals vì nó không phù hợp cho phân loại
