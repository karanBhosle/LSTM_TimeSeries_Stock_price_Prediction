
# **LSTM_TimeSeries_Stock_price_Prediction**

### **Project Overview:**
This project uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical data. The model predicts the closing price of Apple Inc. (AAPL) stock for a given time period using data from Yahoo Finance. The project covers data preprocessing, LSTM model construction, training, evaluation, and visualization of predictions.

The companion notebook (`LSTM_TimeSeries_Stock_price_Prediction/LSTM_TimeSeries_Stock_price_Prediction.ipynb`) now includes **Learning** markdown before every code cell. Those notes explain the purpose of each step in plain language while **preserving original cell outputs** so you can still compare against earlier runs.

### **Why these learnings matter (for data scientists):**

Financial time series are not standard tabular ML problems. Prices are **ordered**, **non-stationary**, and **noisy**. The notebook learnings are structured to build habits that transfer beyond this single ticker:

| Theme | What you practice | Why it matters in real work |
|--------|-------------------|-----------------------------|
| **Reproducible data** | `yfinance` with fixed date ranges | Auditable experiments and fair comparison across model changes |
| **Scaling** | Min–Max normalization | Stable neural network training; understanding train-only fitting vs. leakage |
| **Chronological splits** | 80/20 split without shuffling | Simulates forecasting the future; random splits inflate metrics |
| **Windowing** | `seq_length=100` sequences | Turns raw series into supervised `(X, y)` for LSTMs |
| **Architecture** | Stacked LSTMs + dense head | Balancing capacity vs. overfitting on limited financial signal |
| **Evaluation** | Inverse scaling + test-set plots | Metrics and charts must be in **dollars**, not scaled units |

Mastering this pipeline teaches **time series discipline**: split in time, respect causality, visualize before and after modeling, and treat strong in-sample fit with skepticism on held-out periods. That mindset is essential whether you work in quant research, demand forecasting, IoT sensors, or any domain where tomorrow depends on yesterday—not a random row from the past.

### **Key Features:**
- **Data Acquisition**: Retrieves historical stock data for Apple (AAPL) from Yahoo Finance using the `yfinance` library.
- **Data Preprocessing**: Applies MinMax scaling to normalize stock prices and prepares data for time series forecasting.
- **Educational notebook**: Per-cell **Learning** sections document intent, pitfalls, and best practices.
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

2. **Run the notebook**:
   Open `LSTM_TimeSeries_Stock_price_Prediction/LSTM_TimeSeries_Stock_price_Prediction.ipynb`, read each **Learning** block, then run the code cells in order. Outputs from the original notebook are retained where cells were previously executed.

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

### **Suggested learning path:**
1. Read the notebook intro and each **Learning** markdown cell.
2. Run imports and data download; confirm date range and scaled values.
3. Trace how `create_sequences` defines labels (next-day close).
4. Train the model and compare validation loss across epochs.
5. Interpret the test plot: trend tracking vs. lag and volatility misses.
6. Optional extensions: fit `MinMaxScaler` only on train, add dropout, walk-forward validation, or multivariate features (volume, returns).

### **Code Structure:**
1. **Import Libraries**: Includes libraries for data handling (Pandas, NumPy), visualization (Matplotlib), and model building (TensorFlow/Keras).
2. **Data Download and Preprocessing**: Downloads the stock data, scales it, and creates sequences for the LSTM model.
3. **Model Building**: Defines the LSTM model architecture.
4. **Model Training**: Trains the LSTM model using the training data and validates it with the test data.
5. **Prediction and Visualization**: Visualizes the predicted stock prices against the actual stock prices.

### **Contact Information:**
- **Project Developed by**: Karan Bhosle
- **LinkedIn Profile**: [Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)

Feel free to reach out for questions or collaborations!
