# Car-Data-Analysis-Project
## Project Description
This project involves a detailed analysis of car data to understand the relationship between various attributes such as price, fuel efficiency, aspiration type, and horsepower. Using Python and various data analysis libraries, statistical tests and visualizations are conducted to derive meaningful insights.

## Key Analyses
1. **Correlation Analysis:**
   - Investigated the correlation between car price and fuel efficiency (both city and highway km/l).

2. **Hypothesis Testing on Aspiration Type:**
   - Conducted a Z-test to compare the mean prices of cars with different aspiration types (standard vs. turbo).

3. **Hypothesis Testing on Fuel Type:**
   - Performed a T-test to compare the mean horsepower of cars with different fuel types (gas vs. diesel).

## Results
1. **Correlation Analysis:**
   - The correlation matrix revealed a significant negative correlation between car price and fuel efficiency, indicating that more expensive cars tend to have lower fuel efficiency.

2. **Aspiration Type Hypothesis Test:**
   - The Z-test results indicated a significant difference in mean prices between cars with standard and turbo aspiration, with turbo cars being generally more expensive.

3. **Fuel Type Hypothesis Test:**
   - The T-test results showed no significant difference in mean horsepower between gas and diesel cars.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scipy.stats

## Getting Started

#### Prerequisites
Ensure you have Python installed. You can install the necessary libraries using pip:

```sh
pip install pandas numpy matplotlib seaborn scipy
```

#### Running the Analysis
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/car-data-analysis.git
   cd car-data-analysis
   ```

2. Open the Jupyter Notebook or your preferred Python environment and run the scripts.

### Repository Structure
- `data/`: Contains the dataset file.
- `notebooks/`: Jupyter Notebooks with analysis code.
- `images/`: Contains images used in the README.
- `README.md`: Project documentation.

### Detailed Steps and Code

#### Correlation Analysis
```python
import pandas as pd

# Load data
df = pd.read_csv('data/car_data.csv')

# Calculate correlation matrix
correlation_matrix = df[['price', 'city-km/l', 'highway-km/l']].corr()
print(correlation_matrix)
```

#### Aspiration Type Hypothesis Test
```python
import numpy as np
from scipy.stats import norm

# Extract prices
turbo_prices = np.array(df['price'][df['aspiration'] == 'turbo'].dropna())
standard_prices = np.array(df['price'][df['aspiration'] == 'std'].dropna())

# Z-test function
def two_z_test(sample1, sample2):
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    std1 = np.std(sample1, ddof=1)
    std2 = np.std(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)
    z_stat = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
    return z_stat, p_value

# Perform Z-test
z_stat, p_value = two_z_test(turbo_prices, standard_prices)
print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")
```

#### Fuel Type Hypothesis Test
```python
from scipy.stats import t

# Extract horsepower
diesel_horsepower = np.array(df['horsepower'][df['fuel-type'] == 'diesel'].dropna())
gas_horsepower = np.array(df['horsepower'][df['fuel-type'] == 'gas'].dropna())

# T-test function
def two_sample_t_test(sample1, sample2):
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    std1 = np.std(sample1, ddof=1)
    std2 = np.std(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)
    pooled_std = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    t_stat = (mean1 - mean2) / pooled_std
    df = min(n1, n2) - 1
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=df))
    return t_stat, p_value

# Perform T-test
t_stat, p_value = two_sample_t_test(diesel_horsepower, gas_horsepower)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
```

### Visualizations

#### Correlation Matrix
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

#### Box Plot for Aspiration Type vs. Price
```python
sns.boxplot(x='aspiration', y='price', data=df)
plt.title('Box Plot of Price by Aspiration Type')
plt.show()
```

#### Box Plot for Fuel Type vs. Horsepower
```python
sns.boxplot(x='fuel-type', y='horsepower', data=df)
plt.title('Box Plot of Horsepower by Fuel Type')
plt.show()
```
