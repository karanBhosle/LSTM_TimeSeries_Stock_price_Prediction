
# **LSTM_TimeSeries_Stock_price_Prediction**

### **Project Overview:**
This project uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical data. The model predicts the closing price of Apple Inc. (AAPL) stock for a given time period using data from Yahoo Finance. The project covers data preprocessing, LSTM model construction, training, evaluation, and visualization of predictions.

### **Key Features:**
- **Data Acquisition**: Retrieves historical stock data for Apple (AAPL) from Yahoo Finance using the `yfinance` library.
- **Data Preprocessing**: Applies MinMax scaling to normalize stock prices and prepares data for time series forecasting.
- **Model Architecture**: Utilizes an LSTM-based model for time series prediction.
- **Prediction Visualization**: Compares the predicted stock prices with the actual stock prices.
- **Model Training**: The model is trained on the training data for 10 epochs and then evaluated on the test data.

### **Technologies Used:**
- **yfinance**: To fetch historical stock data from Yahoo Finance.
- **pandas**: For data manipulation and processing.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **TensorFlow/Keras**: To build and train the LSTM model for stock price prediction.
- **scikit-learn**: For MinMax scaling to normalize the stock prices.

### **Getting Started:**
1. **Install Dependencies**:
   Install the necessary Python packages by running the following command:
   ```bash
   pip install yfinance tensorflow numpy pandas scikit-learn matplotlib
   ```

2. **Run the Code**:
   The script fetches the stock data for Apple (AAPL) from Yahoo Finance, preprocesses the data, and builds an LSTM model to predict future stock prices.

### **Model Architecture:**
The LSTM model consists of:
1. **LSTM Layer (50 units)**: The first LSTM layer processes the sequential data and returns sequences for the next layer.
2. **LSTM Layer (50 units)**: A second LSTM layer for deeper feature extraction.
3. **Dense Layer (1 unit)**: The output layer with 1 unit to predict the stock price.

### **Training the Model:**
- **Optimizer**: Adam optimizer is used to minimize the loss.
- **Loss Function**: Mean Squared Error (MSE) is used for regression tasks.
- **Epochs**: The model is trained for 10 epochs, with a batch size of 32.
- **Validation**: The model is validated on the test set during training.

### **Evaluating the Model:**
- **Model Predictions**: Predictions are made for both the training and testing datasets.
- **Inverse Scaling**: Predictions are inverted from the scaled data to the actual stock prices.
- **Visualization**: The actual vs. predicted stock prices are visualized using a plot.

### **Code Structure:**
1. **Import Libraries**: Includes libraries for data handling (Pandas, NumPy), visualization (Matplotlib), and model building (TensorFlow/Keras).
2. **Data Download and Preprocessing**: Downloads the stock data, scales it, and creates sequences for the LSTM model.
3. **Model Building**: Defines the LSTM model architecture.
4. **Model Training**: Trains the LSTM model using the training data and validates it with the test data.
5. **Prediction and Visualization**: Visualizes the predicted stock prices against the actual stock prices.

### **Running the Code:**
1. **Step 1**: Install the necessary dependencies and import the libraries.
2. **Step 2**: Download historical stock data for Apple (AAPL).
3. **Step 3**: Preprocess the data by normalizing the stock prices.
4. **Step 4**: Build and train the LSTM model.
5. **Step 5**: Evaluate the model and visualize the predicted vs. actual stock prices.

### **Contact Information:**
- **Project Developed by**: Karan Bhosle
- **LinkedIn Profile**: [Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)

Feel free to reach out for questions or collaborations!
