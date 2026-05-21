# SalesAnalysis

A full data science pipeline built on a retail sales dataset. It cleans the data, explores patterns, builds a machine learning model, and automates reporting.

---

## What the ML Results Mean

The model's job is simple: **look at an order and predict whether it will make money or lose money.**

### Accuracy — 79.5%
Out of every 100 orders, the model predicted correctly about 80 times. That sounds good, but read on.

### The Confusion Matrix (the honest picture)

|  | Predicted: Lose Money | Predicted: Make Money |
|---|---|---|
| **Actually Lost Money** | 1 ✅ | 40 ❌ |
| **Actually Made Money** | 1 ❌ | 158 ✅ |

The model almost never catches orders that will lose money — it only got 1 out of 41 right. Why? Because 79% of all orders are profitable, so the model learned to just predict "profitable" almost every time. This is called **class imbalance** and is a known problem to fix in the next iteration.

### F1 Score — 0.885
A combined measure of how precise and complete the predictions are. Closer to 1.0 is better. The high score here is misleading for the same reason as accuracy — it's inflated by how often the model correctly predicts profitable orders.

### Cross-Validation — 79.4% ± 0.3%
The model was tested 5 different times on different slices of the data and scored consistently each time. This confirms the model isn't getting lucky on one particular test — it performs the same way across the board.

### Top Features (what actually drives the prediction)
1. **Sales** — how much money the order is worth
2. **Revenue Per Item** — sales divided by quantity ordered
3. **Quantity** — number of items in the order
4. **Order Month** — some months are more profitable than others

Higher value orders tend to be more profitable. The model picked up on this pattern correctly.

---

## Files
- `sales_pipeline.py` — full pipeline: cleaning, EDA, feature engineering, ML model
- `superstore.csv` — the dataset (1,000 retail orders)
- `eda_charts.png` — 9 charts exploring the data
- `ml_results.png` — confusion matrix, feature importance, model scores
