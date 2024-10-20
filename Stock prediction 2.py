# Step: Display Complete Stock Data
# Optionally set display options to show all rows and columns
pd.set_option('display.max_rows', None)  # None means no limit on rows
pd.set_option('display.max_columns', None)  # None means no limit on columns

print("\n--- Complete Stock Data ---\n")
print(combined_data)
# Calculate 100-day SMA for all stocks
for symbol in stock_symbols:
    combined_data[f'{symbol}_SMA_100'] = combined_data[symbol].rolling(window=100).mean()
# Step: Plot Close Prices and 100-Day SMA for All Stocks
plt.figure(figsize=(12, 6))
for symbol in stock_symbols:
    plt.plot(combined_data.index, combined_data[symbol], label=f'{symbol} Close Price', alpha=0.5)  # Close Price
    plt.plot(combined_data.index, combined_data[f'{symbol}_SMA_100'], label=f'{symbol} 100-Day SMA', linestyle='--')  # 100-Day SMA

plt.title('Stock Prices and 100-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
# Calculate 200-day SMA for all stocks
for symbol in stock_symbols:
    combined_data[f'{symbol}_SMA_200'] = combined_data[symbol].rolling(window=200).mean()
# Step: Plot Close Prices and 200-Day SMA for All Stocks
plt.figure(figsize=(12, 6))
for symbol in stock_symbols:
    plt.plot(combined_data.index, combined_data[symbol], label=f'{symbol} Close Price', alpha=0.5)  # Close Price
    plt.plot(combined_data.index, combined_data[f'{symbol}_SMA_200'], label=f'{symbol} 200-Day SMA', linestyle='--')  # 200-Day SMA

plt.title('Stock Prices and 200-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
# Step: Plot Close Prices, 100-Day SMA, and 200-Day SMA for All Stocks
plt.figure(figsize=(12, 6))
for symbol in stock_symbols:
    # Plot Close Price
    plt.plot(combined_data.index, combined_data[symbol], label=f'{symbol} Close Price', alpha=0.5)
    # Plot 100-Day SMA
    plt.plot(combined_data.index, combined_data[f'{symbol}_SMA_100'], label=f'{symbol} 100-Day SMA', linestyle='--')
    # Plot 200-Day SMA
    plt.plot(combined_data.index, combined_data[f'{symbol}_SMA_200'], label=f'{symbol} 200-Day SMA', linestyle='-.')

plt.title('Stock Prices with 100-Day and 200-Day SMAs')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
print(combined_data.shape)
from sklearn.model_selection import train_test_split

# Assuming 'combined_data' is your DataFrame with all stock features and target
# Set 'primary_stock' as the target (i.e., the stock you are predicting)

# Features (X): You can drop columns you don't want to include as features
X = combined_data.drop([primary_stock], axis=1).values

# Target (y): The stock's closing price (primary stock)
y = combined_data[primary_stock].values

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # shuffle=False because it's time-series data

# Display the shape of training and testing datasets
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
# Import pandas for displaying as DataFrame
import pandas as pd

# Convert X_test and y_test to DataFrame for easier viewing
X_test_df = pd.DataFrame(X_test, columns=combined_data.drop([primary_stock], axis=1).columns)
y_test_df = pd.DataFrame(y_test, columns=[primary_stock])

# Display the first 5 rows of X_test (features)
print("X_test (features) head:")
print(X_test_df.head())

# Display the first 5 rows of y_test (target/labels)
print("\ny_test (target) head:")
print(y_test_df.head())
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Assuming combined_data is your DataFrame containing both features and the target
# Identify numeric columns
numeric_columns = combined_data.select_dtypes(include=['float64', 'int64']).columns

# Split your data into features (X) and target (y)
X = combined_data[numeric_columns]
y = combined_data[primary_stock]  # Replace primary_stock with your actual target column

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the training and testing feature sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easier interpretation
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_columns)

# Display the first 5 rows of the scaled training data
print("X_train_scaled (features) head:")
print(X_train_scaled_df.head())

# Display the first 5 rows of the scaled testing data
print("\nX_test_scaled (features) head:")
print(X_test_scaled_df.head())
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming y_train and y_test are your target variables
# Initialize the MinMaxScaler for the target if it is numeric
if y_train.dtype.kind in 'bifc':  # Check if the target is numeric
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale the target
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    # Convert scaled target arrays back to DataFrames for easier interpretation
    y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=[primary_stock])
    y_test_scaled_df = pd.DataFrame(y_test_scaled, columns=[primary_stock])
else:
    # If the target is not numeric, no scaling is needed
    y_train_scaled_df = y_train.reset_index(drop=True)
    y_test_scaled_df = y_test.reset_index(drop=True)

# Display the first 5 rows of the scaled training target
print("y_train_scaled (target) head:")
print(y_train_scaled_df.head())

# Display the first 5 rows of the scaled testing target
print("\ny_test_scaled (target) head:")
print(y_test_scaled_df.head())
import numpy as np

# Assuming you already have X_train and y_train from previous steps (unscaled or scaled)
X_train_array = np.array(X_train)  # Convert X_train to a NumPy array
y_train_array = np.array(y_train)  # Convert y_train to a NumPy array

# Define the window size (100 days of historical data)
window_size = 100

# Create empty lists to store the sequences
X_train_windowed = []
y_train_windowed = []

# Loop through the data to append the sequences
for i in range(window_size, len(X_train_array)):
    # Append the previous 100 days of features to X_train_windowed
    X_train_windowed.append(X_train_array[i - window_size:i])

    # Append the corresponding target value (the next day's stock price) to y_train_windowed
    y_train_windowed.append(y_train_array[i])

# Convert the lists to NumPy arrays
X_train_windowed = np.array(X_train_windowed)
y_train_windowed = np.array(y_train_windowed)

# Check the shape of the new training data
print(f'X_train_windowed shape: {X_train_windowed.shape}')
print(f'y_train_windowed shape: {y_train_windowed.shape}')
import numpy as np
import pandas as pd

# Example: Create a sample DataFrame
data = {
    'column1': np.random.rand(200),  # Replace with your actual data
    'column2': np.random.rand(200)
}
data_df = pd.DataFrame(data)

# Convert the DataFrame to a NumPy array
data_training_array = data_df.values  # This will create a NumPy array from the DataFrame
