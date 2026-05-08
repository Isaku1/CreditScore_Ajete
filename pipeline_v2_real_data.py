"""
Credit Score Classification v2 — Customer-Level Aggregation
XGBoost + Random Forest + Stacking Ensemble
Optuna TPE + SMOTEENN
"""
import numpy as np, pandas as pd, warnings, time, json, pickle
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
import optuna; from optuna.samplers import TPESampler
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
T0 = time.time()

print("╔" + "═"*68 + "╗")
print("║  CREDIT SCORE v2 — CUSTOMER-LEVEL AGGREGATION                     ║")
print("║  XGBoost + Random Forest + Stacking Ensemble                      ║")
print("╚" + "═"*68 + "╝\n")

# ===================== 1. LOAD & CLEAN =====================
print("=" * 70)
print("1. DATA LOADING & PER-CUSTOMER AGGREGATION")
print("=" * 70)

df = pd.read_csv('/mnt/user-data/uploads/train.csv', low_memory=False)
print(f"Raw: {df.shape} | {df['Customer_ID'].nunique()} unique customers × 8 months")

# Clean numeric columns FIRST (before aggregation)
num_to_clean = ['Age','Annual_Income','Num_of_Loan','Num_of_Delayed_Payment',
                'Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly',
                'Monthly_Balance']
for col in num_to_clean:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.\-]','',regex=True), errors='coerce')

df.loc[(df['Age'] < 0) | (df['Age'] > 120), 'Age'] = np.nan

# Parse Credit_History_Age before aggregation
def parse_hist(v):
    if pd.isna(v) or str(v) in ('nan','NA'): return np.nan
    try:
        s = str(v)
        y = int(s.split('Year')[0].strip()) if 'Year' in s else 0
        m = 0
        if 'Month' in s:
            m = int(s.split('and')[-1].strip().split('Month')[0].strip())
        return y * 12 + m
    except: return np.nan

df['Credit_History_Months'] = df['Credit_History_Age'].apply(parse_hist)

# Clean categoricals
df['Credit_Mix_clean'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2, '_': np.nan})
df['Payment_of_Min_Amount_clean'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0, 'NM': 0})
df['Payment_Behaviour_clean'] = df['Payment_Behaviour'].apply(lambda x: x if x != '!@9#%8' else np.nan)

# ---- AGGREGATE BY CUSTOMER ----
# Numeric: median (robust to outliers + corrupted values)
# Categorical: mode
# Credit_Score: mode (most frequent label across 8 months)

numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Months',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
    'Credit_Mix_clean', 'Payment_of_Min_Amount_clean']

agg_dict = {col: 'median' for col in numeric_cols}

# For categorical: take mode
def safe_mode(s):
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else np.nan

# Aggregate
agg_num = df.groupby('Customer_ID')[numeric_cols].median()

# Occupation: mode per customer
agg_occ = df.groupby('Customer_ID')['Occupation'].agg(safe_mode)

# Payment Behaviour: mode per customer
agg_pb = df.groupby('Customer_ID')['Payment_Behaviour_clean'].agg(safe_mode)

# Type_of_Loan: count unique types across all months
agg_loans = df.groupby('Customer_ID')['Type_of_Loan'].agg(
    lambda x: len(set(','.join(x.dropna().astype(str)).split(','))) if x.notna().any() else 0
)

# Credit_Score: mode
agg_target = df.groupby('Customer_ID')['Credit_Score'].agg(safe_mode)

# Combine
cust_df = agg_num.copy()
cust_df['Occupation'] = agg_occ
cust_df['Payment_Behaviour'] = agg_pb
cust_df['Num_Loan_Types'] = agg_loans
cust_df['Credit_Score'] = agg_target

# Also add variance features (captures instability across months)
for col in ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Utilization_Ratio', 'Monthly_Balance']:
    cust_df[f'{col}_std'] = df.groupby('Customer_ID')[col].std().fillna(0)

# Max delay (worst month behavior)
cust_df['Max_Delay'] = df.groupby('Customer_ID')['Delay_from_due_date'].max()
cust_df['Max_Delayed_Payments'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].max()

cust_df = cust_df.reset_index(drop=True)
print(f"Aggregated: {cust_df.shape} (1 row per customer)")
print(f"Target: {cust_df['Credit_Score'].value_counts().to_dict()}")

# ===================== 2. ENCODE & ENGINEER =====================
print(f"\n{'='*70}")
print("2. ENCODING & FEATURE ENGINEERING")
print("=" * 70)

# Occupation one-hot
cust_df['Occupation'] = cust_df['Occupation'].fillna('Unknown')
occ = pd.get_dummies(cust_df['Occupation'], prefix='Occ', drop_first=True)
cust_df = pd.concat([cust_df, occ], axis=1)
cust_df.drop(columns=['Occupation'], inplace=True)

# Payment Behaviour one-hot
cust_df['Payment_Behaviour'] = cust_df['Payment_Behaviour'].fillna('Unknown')
pb = pd.get_dummies(cust_df['Payment_Behaviour'], prefix='PB', drop_first=True)
cust_df = pd.concat([cust_df, pb], axis=1)
cust_df.drop(columns=['Payment_Behaviour'], inplace=True)

# Encode target
le = LabelEncoder()
cust_df['Credit_Score'] = le.fit_transform(cust_df['Credit_Score'])
print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Drop remaining objects
for c in cust_df.select_dtypes(include='object').columns:
    cust_df.drop(columns=[c], inplace=True)

# Impute
for c in cust_df.columns:
    if cust_df[c].isnull().any():
        cust_df[c].fillna(cust_df[c].median(), inplace=True)

X = cust_df.drop(columns=['Credit_Score'])
y = cust_df['Credit_Score']

# Feature engineering
X['Debt_to_Income'] = X['Outstanding_Debt'] / (X['Annual_Income'] + 1)
X['EMI_to_Income'] = (X['Total_EMI_per_month'] * 12) / (X['Annual_Income'] + 1)
X['Delay_Rate'] = X['Num_of_Delayed_Payment'] / (X['Credit_History_Months'] + 1)
X['Investment_Ratio'] = (X['Amount_invested_monthly'] * 12) / (X['Annual_Income'] + 1)
X['Util_x_Delay'] = X['Credit_Utilization_Ratio'] * X['Delay_from_due_date']
X['Income_per_Loan'] = X['Annual_Income'] / (X['Num_of_Loan'] + 1)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print(f"Features: {X.shape[1]}")

# ===================== 3. SPLIT, SCALE, SMOTEENN =====================
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
print(f"Train: {len(y_tr):,}  Test: {len(y_te):,}")

print(f"\n{'='*70}")
print("3. SMOTEENN")
print("=" * 70)
print(f"Before: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
sm = SMOTEENN(random_state=42, n_jobs=-1)
X_tr_r, y_tr_r = sm.fit_resample(X_tr_sc, y_tr)
print(f"After:  {dict(zip(*np.unique(y_tr_r, return_counts=True)))}")
print(f"Resampled: {len(y_tr_r):,}")

pickle.dump({'X_tr_r':X_tr_r,'y_tr_r':y_tr_r,'X_te_sc':X_te_sc,
    'y_te':y_te,'le':le,'feat_names':list(X.columns)}, open('v2_data.pkl','wb'))
print(f"\nStage done: {time.time()-T0:.0f}s")
