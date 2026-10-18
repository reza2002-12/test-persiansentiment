import pandas as pd
from pathlib import Path

# مسیر پوشه داده‌ها
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# خواندن فایل‌های دیتاست
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print("=" * 50)
print("اطلاعات دیتاست")
print("=" * 50)

print(f"\nتعداد داده‌های Train: {len(train)}")
print(f"تعداد داده‌های Validation: {len(val)}")
print(f"تعداد داده‌های Test: {len(test)}")

print("\nستون‌های دیتاست:")
print(train.columns.tolist())

print("\nاطلاعات Train:")
print(train.info())

print("\nتعداد هر برچسب در Train:")
print(train["label"].value_counts().sort_index())

print("\nچند نمونه از داده‌ها:")
print(train.head(10))