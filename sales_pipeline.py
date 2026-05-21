# ============================================================
#  SALES DATA SCIENCE PIPELINE
#  Full walkthrough: Pandas → EDA → ML → Evaluation
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)

# ─────────────────────────────────────────────
# STEP 1: LOAD & EXPLORE THE DATA
# ─────────────────────────────────────────────
# WHY: Before touching data, understand what you have.
# This is always the first thing a data scientist does.

print("=" * 60)
print("STEP 1: LOAD & EXPLORE")
print("=" * 60)

# pd.read_csv() loads a CSV file into a DataFrame
# A DataFrame is like an Excel table — rows and columns
df = pd.read_csv('superstore.csv')

# .shape returns (number of rows, number of columns)
print(f"\nDataset shape: {df.shape}")
print(f"→ We have {df.shape[0]} orders and {df.shape[1]} columns\n")

# .head() shows the first 5 rows — always run this first
print("First 5 rows:")
print(df.head())

# .info() shows column names, data types, and non-null counts
# This tells you which columns have missing data at a glance
print("\nDataset info:")
print(df.info())

# .describe() gives you statistics for every numeric column
# mean, std, min, max, percentiles (25%, 50%, 75%)
print("\nNumeric statistics:")
print(df.describe().round(2))

# .isnull().sum() counts missing values per column
# This is how you find where data is incomplete
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# STEP 2: CLEAN THE DATA
# ─────────────────────────────────────────────
# WHY: Real data is always messy. Cleaning is 70% of a
# data scientist's job. Clean data = trustworthy results.

print("\n" + "=" * 60)
print("STEP 2: CLEAN THE DATA")
print("=" * 60)

# Always work on a copy so you don't modify the original
# This is good practice — you can always go back to df
df_clean = df.copy()

# Convert date columns from strings to actual datetime objects
# WHY: So we can do math on dates (e.g. how many days between orders)
df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'])
df_clean['Ship_Date'] = pd.to_datetime(df_clean['Ship_Date'])
print("\nConverted Order_Date and Ship_Date to datetime ✓")

# Create a new column: how many days to ship?
# WHY: Feature engineering — creating useful info from existing columns
df_clean['Days_to_Ship'] = (df_clean['Ship_Date'] - df_clean['Order_Date']).dt.days
print(f"Created Days_to_Ship column. Range: {df_clean['Days_to_Ship'].min()} to {df_clean['Days_to_Ship'].max()} days")

# Extract year and month from Order_Date
# WHY: Useful for trend analysis and as ML features
df_clean['Order_Year'] = df_clean['Order_Date'].dt.year
df_clean['Order_Month'] = df_clean['Order_Date'].dt.month
print("Extracted Order_Year and Order_Month ✓")

# Fill missing Discount values with 0 (no discount)
# WHY: Missing discount most likely means no discount was applied
df_clean['Discount'] = df_clean['Discount'].fillna(0)
print(f"\nFilled {df['Discount'].isnull().sum()} missing Discount values with 0 ✓")

# Fill missing Profit values with the median
# WHY: Median is better than mean when you might have outliers
# (a few huge profits/losses would skew the mean)
median_profit = df_clean['Profit'].median()
df_clean['Profit'] = df_clean['Profit'].fillna(median_profit)
print(f"Filled {df['Profit'].isnull().sum()} missing Profit values with median (${median_profit:.2f}) ✓")

# drop_duplicates() removes any completely identical rows
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"\nRemoved {before - len(df_clean)} duplicate rows ✓")

# Verify: no more missing values
print(f"\nMissing values after cleaning: {df_clean.isnull().sum().sum()}")
print(f"Final clean dataset shape: {df_clean.shape}")


# ─────────────────────────────────────────────
# STEP 3: ANALYZE & VISUALIZE
# ─────────────────────────────────────────────
# WHY: EDA (Exploratory Data Analysis) helps you understand
# patterns before building models. You always visualize first.

print("\n" + "=" * 60)
print("STEP 3: ANALYZE & VISUALIZE")
print("=" * 60)

# --- KEY BUSINESS QUESTIONS ---

# groupby() groups rows by a column, then you aggregate
# Here: total Sales per Category
sales_by_cat = df_clean.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print("\nTotal Sales by Category:")
print(sales_by_cat.round(2))

# Multiple aggregations at once with .agg()
# We want both sum AND mean for Sales and Profit
region_summary = df_clean.groupby('Region').agg(
    Total_Sales=('Sales', 'sum'),
    Avg_Profit=('Profit', 'mean'),
    Order_Count=('Order_ID', 'count')
).round(2)
print("\nRegion Summary:")
print(region_summary)

# value_counts() shows frequency of each unique value
print("\nShip Mode Distribution:")
print(df_clean['Ship_Mode'].value_counts())

# Profitable vs not — we'll use this as our ML target
df_clean['Is_Profitable'] = (df_clean['Profit'] > 0).astype(int)
profit_rate = df_clean['Is_Profitable'].mean() * 100
print(f"\nProfitable orders: {profit_rate:.1f}%")
print(f"Unprofitable orders: {100-profit_rate:.1f}%")


# ─────────────────────────────────────────────
# STEP 3B: CREATE VISUALIZATIONS
# ─────────────────────────────────────────────

print("\nGenerating visualizations...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Color palette
colors = ['#2E75B6', '#70AD47', '#ED7D31', '#FFC000']
cat_colors = {'Technology': '#2E75B6', 'Furniture': '#ED7D31', 'Office Supplies': '#70AD47'}

# --- Chart 1: Sales by Category (Bar) ---
ax1 = fig.add_subplot(gs[0, 0])
sales_by_cat.plot(kind='bar', ax=ax1, color=[cat_colors[c] for c in sales_by_cat.index], edgecolor='white')
ax1.set_title('Total Sales by Category', fontweight='bold', fontsize=11)
ax1.set_xlabel('')
ax1.set_ylabel('Total Sales ($)')
ax1.tick_params(axis='x', rotation=15)
for i, v in enumerate(sales_by_cat):
    ax1.text(i, v + 1000, f'${v:,.0f}', ha='center', fontsize=8, fontweight='bold')

# --- Chart 2: Profit by Region (Bar) ---
ax2 = fig.add_subplot(gs[0, 1])
avg_profit_region = df_clean.groupby('Region')['Profit'].mean()
bars = ax2.bar(avg_profit_region.index, avg_profit_region.values,
               color=['#2ECC71' if v > 0 else '#E74C3C' for v in avg_profit_region.values],
               edgecolor='white')
ax2.set_title('Avg Profit by Region', fontweight='bold', fontsize=11)
ax2.set_ylabel('Avg Profit ($)')
ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

# --- Chart 3: Profitable vs Not (Pie) ---
ax3 = fig.add_subplot(gs[0, 2])
profit_counts = df_clean['Is_Profitable'].value_counts()
ax3.pie(profit_counts.values,
        labels=['Profitable', 'Unprofitable'],
        colors=['#2ECC71', '#E74C3C'],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax3.set_title('Profitable vs Unprofitable Orders', fontweight='bold', fontsize=11)

# --- Chart 4: Sales Distribution (Histogram) ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df_clean['Sales'], bins=40, color='#2E75B6', edgecolor='white', alpha=0.8)
ax4.set_title('Sales Distribution', fontweight='bold', fontsize=11)
ax4.set_xlabel('Sales ($)')
ax4.set_ylabel('Count')
ax4.axvline(df_clean['Sales'].median(), color='#E74C3C', linestyle='--', linewidth=2, label=f'Median: ${df_clean["Sales"].median():.0f}')
ax4.legend(fontsize=8)

# --- Chart 5: Sales by Segment (Box plot) ---
ax5 = fig.add_subplot(gs[1, 1])
segments_data = [df_clean[df_clean['Segment'] == s]['Sales'].values for s in df_clean['Segment'].unique()]
bp = ax5.boxplot(segments_data, labels=df_clean['Segment'].unique(), patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_title('Sales Distribution by Segment', fontweight='bold', fontsize=11)
ax5.set_ylabel('Sales ($)')
ax5.tick_params(axis='x', rotation=10)

# --- Chart 6: Monthly Sales Trend ---
ax6 = fig.add_subplot(gs[1, 2])
monthly = df_clean.groupby(['Order_Year', 'Order_Month'])['Sales'].sum().reset_index()
monthly['Period'] = monthly['Order_Year'].astype(str) + '-' + monthly['Order_Month'].astype(str).str.zfill(2)
ax6.plot(range(len(monthly)), monthly['Sales'], color='#2E75B6', linewidth=2, marker='o', markersize=3)
ax6.set_title('Monthly Sales Trend', fontweight='bold', fontsize=11)
ax6.set_ylabel('Total Sales ($)')
ax6.set_xlabel('Time →')
ax6.tick_params(axis='x', labelbottom=False)
ax6.fill_between(range(len(monthly)), monthly['Sales'], alpha=0.15, color='#2E75B6')

# --- Chart 7: Discount vs Profit (Scatter) ---
ax7 = fig.add_subplot(gs[2, 0])
scatter_colors = df_clean['Is_Profitable'].map({1: '#2ECC71', 0: '#E74C3C'})
ax7.scatter(df_clean['Discount'], df_clean['Profit'],
            c=scatter_colors, alpha=0.4, s=15)
ax7.set_title('Discount vs Profit', fontweight='bold', fontsize=11)
ax7.set_xlabel('Discount Rate')
ax7.set_ylabel('Profit ($)')
ax7.axhline(0, color='black', linewidth=0.8, linestyle='--')

# --- Chart 8: Top Sub-Categories by Sales ---
ax8 = fig.add_subplot(gs[2, 1])
top_sub = df_clean.groupby('Sub_Category')['Sales'].sum().sort_values(ascending=True).tail(8)
top_sub.plot(kind='barh', ax=ax8, color='#2E75B6', edgecolor='white')
ax8.set_title('Top Sub-Categories by Sales', fontweight='bold', fontsize=11)
ax8.set_xlabel('Total Sales ($)')

# --- Chart 9: Days to Ship by Ship Mode ---
ax9 = fig.add_subplot(gs[2, 2])
ship_days = df_clean.groupby('Ship_Mode')['Days_to_Ship'].mean().sort_values()
ship_days.plot(kind='bar', ax=ax9, color=colors[:len(ship_days)], edgecolor='white')
ax9.set_title('Avg Days to Ship by Mode', fontweight='bold', fontsize=11)
ax9.set_ylabel('Avg Days')
ax9.tick_params(axis='x', rotation=15)

fig.suptitle('Superstore Sales — Exploratory Data Analysis', 
             fontsize=16, fontweight='bold', y=1.01, color='#1F3864')

plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
plt.close()
print("EDA charts saved ✓")


# ─────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────
# WHY: Raw data isn't always what the model needs.
# Feature engineering = creating better inputs for the model.

print("\n" + "=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

# Revenue per item (Sales divided by Quantity)
df_clean['Revenue_Per_Item'] = df_clean['Sales'] / df_clean['Quantity']
print("Created Revenue_Per_Item ✓")

# Is there a discount or not? (Binary)
df_clean['Has_Discount'] = (df_clean['Discount'] > 0).astype(int)
print("Created Has_Discount ✓")

# Is it a high value order? (Above median sales)
median_sales = df_clean['Sales'].median()
df_clean['High_Value_Order'] = (df_clean['Sales'] > median_sales).astype(int)
print(f"Created High_Value_Order (threshold: ${median_sales:.2f}) ✓")

print(f"\nFeature engineering complete. Dataset now has {df_clean.shape[1]} columns")


# ─────────────────────────────────────────────
# STEP 5: BUILD THE ML MODEL
# ─────────────────────────────────────────────
# GOAL: Predict whether an order will be PROFITABLE or not
# This is a CLASSIFICATION problem (output is 0 or 1)
# WHY: If Corsair could predict unprofitable orders early,
# they could adjust pricing or strategy proactively.

print("\n" + "=" * 60)
print("STEP 5: BUILD THE ML MODEL")
print("=" * 60)
print("\nGoal: Predict whether an order will be profitable (1) or not (0)")

# Select features (inputs) and target (what we want to predict)
# We deliberately exclude columns that would "leak" the answer
# e.g. Profit itself tells us if it's profitable — that's cheating!
feature_cols = [
    'Category', 'Sub_Category', 'Region', 'Segment', 'Ship_Mode',  # categorical
    'Sales', 'Quantity', 'Discount', 'Days_to_Ship',                # numerical
    'Revenue_Per_Item', 'Has_Discount', 'Order_Month'               # engineered
]

X = df_clean[feature_cols]   # Features (inputs to the model)
y = df_clean['Is_Profitable'] # Target (what we want to predict)

print(f"\nFeatures used: {feature_cols}")
print(f"Target: Is_Profitable")
print(f"X shape: {X.shape} | y shape: {y.shape}")
print(f"Class balance: {y.value_counts().to_dict()}")

# Split into train and test sets
# test_size=0.2 means 20% held back for testing (200 rows)
# random_state=42 means the same split every time (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} rows")
print(f"Test size:  {X_test.shape[0]} rows")

# Identify which columns are numerical and which are categorical
# We need to handle them differently in preprocessing
numerical_cols = ['Sales', 'Quantity', 'Discount', 'Days_to_Ship',
                  'Revenue_Per_Item', 'Has_Discount', 'Order_Month']
categorical_cols = ['Category', 'Sub_Category', 'Region', 'Segment', 'Ship_Mode']

# Numerical pipeline:
# 1. SimpleImputer fills any remaining missing values with the median
# 2. StandardScaler normalizes values to mean=0, std=1
#    WHY: Models work better when features are on the same scale
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline:
# 1. SimpleImputer fills missing with most frequent value
# 2. OneHotEncoder converts categories to 0/1 columns
#    e.g. Region "West" → [1,0,0,0], "East" → [0,1,0,0]
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer applies different pipelines to different columns
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_cols),
    ('categorical', categorical_pipeline, categorical_cols)
])

# Full pipeline: preprocessing + model in one clean object
# WHY Pipeline: prevents data leakage, makes deployment cleaner,
# and lets you swap models easily
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,   # 100 decision trees in the forest
        max_depth=10,       # max depth of each tree (prevents overfitting)
        random_state=42
    ))
])

print("\nTraining the Random Forest model...")
model_pipeline.fit(X_train, y_train)
print("Model trained! ✓")


# ─────────────────────────────────────────────
# STEP 6: EVALUATE THE MODEL
# ─────────────────────────────────────────────
# WHY: Training accuracy means nothing. You need to know
# how well the model performs on data it has NEVER seen.

print("\n" + "=" * 60)
print("STEP 6: EVALUATE THE MODEL")
print("=" * 60)

# Predict on the test set (data the model has never seen)
y_pred = model_pipeline.predict(X_test)

# Accuracy: % of predictions that were correct
acc = accuracy_score(y_test, y_pred)

# F1 Score: balance between precision and recall
# Better metric than accuracy when classes are imbalanced
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy:  {acc:.1%}")
print(f"F1 Score:  {f1:.3f}")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Unprofitable', 'Profitable']))

# Cross-validation: test on 5 different train/test splits
# WHY: More reliable than a single split — averages out luck
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation (5-fold):")
print(f"  Scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives  (correctly predicted unprofitable): {cm[0,0]}")
print(f"  False Positives (predicted profitable, was not):    {cm[0,1]}")
print(f"  False Negatives (predicted unprofitable, was):      {cm[1,0]}")
print(f"  True Positives  (correctly predicted profitable):   {cm[1,1]}")

# Feature Importance: which features help the model most?
# Only works directly on tree-based models, not the pipeline wrapper
# So we extract the model and get feature names from the preprocessor
feature_names_num = numerical_cols
feature_names_cat = model_pipeline.named_steps['preprocessor']\
    .named_transformers_['categorical']['encoder']\
    .get_feature_names_out(categorical_cols).tolist()
all_feature_names = feature_names_num + feature_names_cat

importances = model_pipeline.named_steps['classifier'].feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(10)

print("\nTop 10 Most Important Features:")
print(feat_importance_df.to_string(index=False))


# ─────────────────────────────────────────────
# STEP 7: VISUALIZE RESULTS
# ─────────────────────────────────────────────

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor('#F8F9FA')

# Chart 1: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Unprofitable', 'Profitable'],
            yticklabels=['Unprofitable', 'Profitable'],
            ax=axes[0], linewidths=1)
axes[0].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Chart 2: Feature Importance
feat_importance_df.plot(kind='barh', x='Feature', y='Importance',
                        ax=axes[1], color='#2E75B6', edgecolor='white', legend=False)
axes[1].set_title('Top 10 Feature Importances', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

# Chart 3: Model Metrics Summary
metrics = ['Accuracy', 'F1 Score', 'CV Mean']
values = [acc, f1, cv_scores.mean()]
bar_colors = ['#2ECC71' if v >= 0.7 else '#E74C3C' for v in values]
bars = axes[2].bar(metrics, values, color=bar_colors, edgecolor='white', width=0.5)
axes[2].set_title('Model Performance Summary', fontweight='bold', fontsize=12)
axes[2].set_ylim(0, 1)
axes[2].axhline(0.7, color='#E74C3C', linestyle='--', linewidth=1.5, label='0.7 threshold')
axes[2].legend(fontsize=9)
for bar, val in zip(bars, values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.1%}', ha='center', fontweight='bold', fontsize=11)

fig2.suptitle('ML Model — Profitability Prediction Results',
              fontsize=14, fontweight='bold', color='#1F3864')
plt.tight_layout()
plt.savefig('ml_results.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
plt.close()
print("\nML result charts saved ✓")


# ─────────────────────────────────────────────
# STEP 8: AUTOMATE — Wrap it all in functions
# ─────────────────────────────────────────────
# WHY: At Corsair, you won't run code once. You'll build
# pipelines that run automatically on new data every day.
# Functions make code reusable, testable, and clean.

print("\n" + "=" * 60)
print("STEP 8: AUTOMATION — MAKING IT REUSABLE")
print("=" * 60)

def load_and_clean(filepath):
    """Load and clean the sales dataset."""
    df = pd.read_csv(filepath)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])
    df['Days_to_Ship'] = (df['Ship_Date'] - df['Order_Date']).dt.days
    df['Order_Year'] = df['Order_Date'].dt.year
    df['Order_Month'] = df['Order_Date'].dt.month
    df['Discount'] = df['Discount'].fillna(0)
    df['Profit'] = df['Profit'].fillna(df['Profit'].median())
    df = df.drop_duplicates()
    return df

def engineer_features(df):
    """Add engineered features to the dataset."""
    df = df.copy()
    df['Revenue_Per_Item'] = df['Sales'] / df['Quantity']
    df['Has_Discount'] = (df['Discount'] > 0).astype(int)
    df['High_Value_Order'] = (df['Sales'] > df['Sales'].median()).astype(int)
    df['Is_Profitable'] = (df['Profit'] > 0).astype(int)
    return df

def generate_report(df):
    """Print a quick business summary report."""
    print("\n📊 BUSINESS SUMMARY REPORT")
    print("-" * 40)
    print(f"Total Orders:     {len(df):,}")
    print(f"Total Revenue:    ${df['Sales'].sum():,.2f}")
    print(f"Total Profit:     ${df['Profit'].sum():,.2f}")
    print(f"Profit Margin:    {(df['Profit'].sum()/df['Sales'].sum()*100):.1f}%")
    print(f"Profitable Orders:{df['Is_Profitable'].mean()*100:.1f}%")
    print(f"\nTop Category:     {df.groupby('Category')['Sales'].sum().idxmax()}")
    print(f"Top Region:       {df.groupby('Region')['Sales'].sum().idxmax()}")
    print(f"Top Segment:      {df.groupby('Segment')['Sales'].sum().idxmax()}")

# Run the full automated pipeline
print("\nRunning full automated pipeline on new data...")
df_auto = load_and_clean('superstore.csv')
df_auto = engineer_features(df_auto)
generate_report(df_auto)

print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETE")
print("=" * 60)
print("""
WHAT YOU JUST BUILT:
  Step 1: Loaded and explored a real dataset (Pandas)
  Step 2: Cleaned missing values, fixed types (Pandas)
  Step 3: Analyzed patterns with groupby + visualizations
  Step 4: Engineered new features from existing columns
  Step 5: Built an ML classification model (Scikit-learn Pipeline)
  Step 6: Evaluated with accuracy, F1, cross-validation, confusion matrix
  Step 7: Visualized model results
  Step 8: Wrapped everything in reusable functions (automation mindset)

CONCEPTS COVERED:
  ✓ pd.read_csv, head, info, describe, isnull
  ✓ fillna, dropna, drop_duplicates, astype, to_datetime
  ✓ groupby, agg, value_counts, apply
  ✓ train_test_split, cross_val_score
  ✓ Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder
  ✓ RandomForestClassifier, fit, predict
  ✓ accuracy_score, f1_score, confusion_matrix, classification_report
  ✓ Feature importance, function-based automation
""")
