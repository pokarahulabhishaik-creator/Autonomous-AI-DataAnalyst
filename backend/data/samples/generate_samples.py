import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_samples():
    np.random.seed(42)
    os.makedirs("backend/data/samples", exist_ok=True)

    # 1. Superstore Sales dataset (1200 rows)
    n = 1200
    categories = {
        'Technology': ['Phones', 'Laptops', 'Accessories', 'Printers'],
        'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
        'Office Supplies': ['Binders', 'Paper', 'Storage', 'Art', 'Fasteners']
    }
    regions = ['North', 'South', 'East', 'West']
    segments = ['Consumer', 'Corporate', 'Home Office']

    records = []
    base_date = datetime(2023, 1, 1)

    for i in range(n):
        order_id = f"ORD-{2023 + i // 600}-{1000 + i}"
        dt = base_date + timedelta(days=int(np.random.uniform(0, 720)))
        cat = np.random.choice(list(categories.keys()), p=[0.35, 0.30, 0.35])
        sub_cat = np.random.choice(categories[cat])
        reg = np.random.choice(regions, p=[0.25, 0.20, 0.30, 0.25])
        seg = np.random.choice(segments, p=[0.50, 0.30, 0.20])
        
        qty = int(np.random.randint(1, 10))
        
        if cat == 'Technology':
            base_price = float(np.random.uniform(150, 900))
        elif cat == 'Furniture':
            base_price = float(np.random.uniform(80, 500))
        else:
            base_price = float(np.random.uniform(15, 120))
            
        discount = float(np.random.choice([0.0, 0.1, 0.2, 0.3, 0.5, 0.7], p=[0.4, 0.25, 0.15, 0.1, 0.07, 0.03]))
        sales = round(base_price * qty * (1.0 - discount), 2)
        
        base_margin = 0.24 if cat == 'Technology' else (0.10 if cat == 'Furniture' else 0.28)
        profit = round(sales * (base_margin - discount * 0.75), 2)
        
        # Inject realistic anomalies
        if i in [42, 312, 788]:
            profit = -round(sales * 1.4, 2)
            
        records.append({
            'order_id': order_id,
            'order_date': dt.strftime('%Y-%m-%d'),
            'customer_id': f"CUST-{np.random.randint(100, 350)}",
            'customer_segment': seg,
            'region': reg,
            'category': cat,
            'sub_category': sub_cat,
            'sales': sales,
            'quantity': qty,
            'discount': discount,
            'profit': profit
        })

    df_superstore = pd.DataFrame(records)
    superstore_path = "backend/data/samples/superstore_sales.csv"
    df_superstore.to_csv(superstore_path, index=False)
    print(f"Created {superstore_path} with {len(df_superstore)} rows")

    # 2. Customer Churn dataset (800 rows)
    churn_records = []
    for i in range(800):
        cid = f"SAAS-{5000 + i}"
        tenure = int(np.random.randint(1, 72))
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.55, 0.25, 0.20])
        monthly = round(float(np.random.uniform(29.99, 129.99)), 2)
        total = round(monthly * tenure * float(np.random.uniform(0.95, 1.05)), 2)
        support_calls = int(np.random.poisson(lam=2.5 if contract == 'Month-to-month' else 1.0))
        payment = np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'])
        
        churn_score = (
            (1.5 if contract == 'Month-to-month' else -0.8) +
            (0.4 * support_calls) +
            (0.015 * (monthly - 60)) -
            (0.04 * tenure)
        )
        churn_prob = 1 / (1 + np.exp(-churn_score))
        churned = 'Yes' if np.random.rand() < churn_prob else 'No'
        
        churn_records.append({
            'customer_id': cid,
            'tenure_months': tenure,
            'contract_type': contract,
            'monthly_charges': monthly,
            'total_charges': total,
            'support_calls': support_calls,
            'payment_method': payment,
            'churn': churned
        })

    df_churn = pd.DataFrame(churn_records)
    churn_path = "backend/data/samples/customer_churn.csv"
    df_churn.to_csv(churn_path, index=False)
    print(f"Created {churn_path} with {len(df_churn)} rows")

if __name__ == "__main__":
    generate_samples()
