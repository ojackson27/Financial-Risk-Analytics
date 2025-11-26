# QuantRisk: Financial Risk Analysis Toolkit

**QuantRisk** is a Python-based library designed for quantitative finance analysis. It provides tools to calculate sophisticated risk metrics, including Drawdowns, Semi-Deviation, and Cornish-Fisher Modified Value at Risk (VaR).

This toolkit accounts for non-normal return distributions (fat tails/skewness), offering more accurate risk assessments than standard Gaussian models.

## Key Features

* **Drawdown Analysis:** Computes wealth index, previous peaks, and percentage drawdowns.
* **Non-Normal Distributions:** Includes custom implementations of Skewness and Kurtosis.
* **Testing for Normality:** Jarque-Bera hypothesis testing.
* **Value at Risk (VaR):**
    * Historic VaR
    * Parametric Gaussian VaR
    * **Cornish-Fisher Modified VaR** (adjusts for skewness and kurtosis)

## Mathematical Theoretical Basis

### Modified Cornish-Fisher VaR
Standard VaR models often underestimate risk by assuming returns follow a normal distribution. **QuantRisk** utilizes the Cornish-Fisher expansion to adjust the $Z$-score based on the asset's specific skewness ($S$) and kurtosis ($K$):

$$z_{cornish} = z_c + \frac{1}{6}(z_c^2 - 1)S + \frac{1}{24}(z_c^3 - 3z_c)(K-3) - \frac{1}{36}(2z_c^3 - 5z_c)S^2$$

Where $z_c$ is the critical value for the normal distribution.

## Installation & Usage

1. Clone the repository.
2. Ensure you have `pandas`, `numpy`, and `scipy` installed.

```python
import pandas as pd
import risk_kit as rk

# Example: Load your own return data
returns = pd.read_csv("data/my_returns.csv", index_col=0, parse_dates=True)

# 1. Check for Normality
is_normal = rk.is_normal(returns)

# 2. Calculate Drawdowns
dd = rk.drawdown(returns["LargeCap"])
dd["Drawdown"].plot()

# 3. Calculate Tail Risk (Modified VaR)
# This accounts for 'Black Swan' events better than standard deviation
cornish_fisher_var = rk.var_gaussian(returns, level=5, modified=True)
print(f"Modified VaR (95%): {cornish_fisher_var * 100:.2f}%")
