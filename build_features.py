"""Dataset-specific cleaning and customer feature engineering for real retail data."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd

from configs import DatasetConfig, get_config


def _log(stage: str) -> None:
    print(f"[features] {stage}", flush=True)


def _excel_engine() -> str:
    """Prefer Calamine for large workbooks; retain openpyxl fallback."""
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return "openpyxl"


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0).clip(lower=0)


def _quality_report(raw_rows: int, clean_rows: int, missing_customers: int, duplicates: int, **extra: Any) -> dict:
    return {
        "raw_rows": int(raw_rows),
        "clean_rows": int(clean_rows),
        "rows_removed": int(raw_rows - clean_rows),
        "missing_customer_rows": int(missing_customers),
        "duplicate_rows_removed": int(duplicates),
        **{key: int(value) if isinstance(value, (int, np.integer)) else value for key, value in extra.items()},
    }


def build_online_retail_features(config: DatasetConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Build RFM and behavior features from the two Online Retail II worksheets.

    Positive, non-cancelled lines define purchase value. Cancellation/negative lines are
    retained only to calculate a per-customer return-line rate.
    """
    config = config or get_config("online_retail_ii")
    workbook = config.data_dir / "online_retail_II.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"Online Retail II workbook not found: {workbook}")

    engine = _excel_engine()
    sheet_names = pd.ExcelFile(workbook, engine=engine).sheet_names
    _log(f"Online Retail II: workbook opened with {len(sheet_names)} sheets")
    column_map = {
        "Customer_ID": "Customer_ID",
        "CustomerID": "Customer_ID",
        "InvoiceDate": "InvoiceDate",
        "Invoice_Date": "InvoiceDate",
        "Invoice": "Invoice",
        "StockCode": "StockCode",
        "Stock_Code": "StockCode",
        "Quantity": "Quantity",
        "Price": "Price",
        "UnitPrice": "Price",
    }
    raw_rows = duplicates = missing_customers = valid_rows = positive_sales_rows = return_rows = 0
    customer_parts: list[pd.DataFrame] = []
    all_line_parts: list[pd.Series] = []
    return_line_parts: list[pd.Series] = []
    product_pairs: list[pd.DataFrame] = []
    for sheet_name in sheet_names:
        _log(f"Online Retail II: reading {sheet_name}")
        data = pd.read_excel(workbook, sheet_name=sheet_name, engine=engine)
        data.columns = [str(column).strip().replace(" ", "_") for column in data.columns]
        data = data.rename(columns={source: target for source, target in column_map.items() if source in data.columns})
        required = {"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer_ID"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Online Retail II sheet '{sheet_name}' is missing: {sorted(missing)}")
        data = data[["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer_ID"]]
        raw_rows += len(data)
        duplicates += int(data.duplicated().sum())
        data = data.drop_duplicates()
        data["Invoice"] = data["Invoice"].astype(str).str.strip()
        data["Customer_ID"] = pd.to_numeric(data["Customer_ID"], errors="coerce")
        data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
        data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
        data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
        missing_customers += int(data["Customer_ID"].isna().sum())
        valid = data.dropna(subset=["Customer_ID", "InvoiceDate", "Quantity", "Price", "Invoice", "StockCode"])
        valid = valid[(valid["Quantity"] != 0) & (valid["Price"] > 0)].copy()
        valid["Customer_ID"] = valid["Customer_ID"].astype("int64").astype(str)
        valid["is_return"] = valid["Invoice"].str.upper().str.startswith("C") | (valid["Quantity"] < 0)
        valid["line_value"] = valid["Quantity"].abs() * valid["Price"]
        sales = valid.loc[~valid["is_return"] & (valid["Quantity"] > 0)]
        valid_rows += len(valid)
        positive_sales_rows += len(sales)
        return_rows += int(valid["is_return"].sum())
        customer_parts.append(sales.groupby("Customer_ID", as_index=False).agg(
            First_Purchase=("InvoiceDate", "min"), Last_Purchase=("InvoiceDate", "max"), Frequency=("Invoice", "nunique"), Monetary=("line_value", "sum"), Quantity=("Quantity", "sum")
        ))
        all_line_parts.append(valid.groupby("Customer_ID").size())
        return_line_parts.append(valid.loc[valid["is_return"]].groupby("Customer_ID").size())
        product_pairs.append(sales[["Customer_ID", "StockCode"]].drop_duplicates())
        del data, valid, sales
        gc.collect()
    if not customer_parts:
        raise ValueError("Online Retail II contained no readable transaction sheets.")
    customer_rollup = pd.concat(customer_parts, ignore_index=True).groupby("Customer_ID", as_index=False).agg(
        First_Purchase=("First_Purchase", "min"), Last_Purchase=("Last_Purchase", "max"), Frequency=("Frequency", "sum"), Monetary=("Monetary", "sum"), Quantity=("Quantity", "sum")
    )
    if customer_rollup.empty:
        raise ValueError("No valid positive Online Retail II transactions remain after cleaning.")
    snapshot = customer_rollup["Last_Purchase"].max().normalize() + pd.Timedelta(days=1)
    unique_products = pd.concat(product_pairs, ignore_index=True).drop_duplicates().groupby("Customer_ID").size()
    all_lines = pd.concat(all_line_parts, axis=1).fillna(0).sum(axis=1)
    return_lines = pd.concat(return_line_parts, axis=1).fillna(0).sum(axis=1)
    purchases = customer_rollup.set_index("Customer_ID")
    purchases["Recency_Days"] = (snapshot - purchases["Last_Purchase"].dt.normalize()).dt.days
    purchases["Tenure_Days"] = (purchases["Last_Purchase"].dt.normalize() - purchases["First_Purchase"].dt.normalize()).dt.days
    purchases["Unique_Products"] = unique_products.reindex(purchases.index).fillna(0)
    purchases["Average_Order_Value"] = _safe_rate(purchases["Monetary"], purchases["Frequency"])
    purchases["Average_Basket_Size"] = _safe_rate(purchases["Quantity"], purchases["Frequency"])
    purchases["Return_Rate"] = _safe_rate(return_lines.reindex(purchases.index).fillna(0), all_lines.reindex(purchases.index).fillna(0))
    purchases = purchases.drop(columns=["First_Purchase", "Last_Purchase", "Quantity"])
    features = purchases.reset_index()
    quality = _quality_report(
        raw_rows, valid_rows, missing_customers, duplicates,
        positive_sales_rows=positive_sales_rows, return_or_cancel_rows=return_rows,
        customer_count=len(features), snapshot_date=str(snapshot.date()),
        rules="Positive non-cancelled lines form sales; cancellation/negative lines only form Return_Rate.",
    )
    return features, quality


def build_olist_features(config: DatasetConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Aggregate Olist at order level before customer-level metrics to avoid double counts."""
    config = config or get_config("olist")
    required_files = {
        "orders": "olist_orders_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
    }
    paths = {key: config.data_dir / name for key, name in required_files.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Olist source files: {missing}")

    _log("Olist: reading orders and customers")
    orders = pd.read_csv(paths["orders"], usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date"])
    customers = pd.read_csv(paths["customers"], usecols=["customer_id", "customer_unique_id"])
    raw_rows = len(orders) + len(customers)

    orders = orders.drop_duplicates("order_id").copy()
    customers = customers.drop_duplicates("customer_id").copy()
    for column in ["order_purchase_timestamp", "order_delivered_customer_date"]:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")
    delivered = orders.loc[
        (orders["order_status"] == "delivered") & orders["order_purchase_timestamp"].notna()
    ].copy()
    delivered_count = len(delivered)
    invalid_order_rows = len(orders) - len(delivered)
    order_core = delivered.merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id", how="inner", validate="many_to_one"
    )
    del orders, customers, delivered

    _log("Olist: aggregating order items")
    products = pd.read_csv(paths["products"], usecols=["product_id", "product_category_name"]).drop_duplicates("product_id")
    items = pd.read_csv(paths["items"], usecols=["order_id", "order_item_id", "product_id", "price", "freight_value"])
    raw_rows += len(products) + len(items)
    enriched_items = items.merge(products, on="product_id", how="left", validate="many_to_one")
    order_items = enriched_items.groupby("order_id", as_index=False).agg(
        item_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        Unique_Products=("product_id", "nunique"),
        Category_Diversity=("product_category_name", "nunique"),
        Item_Count=("order_item_id", "count"),
    )
    order_items["Category_Diversity"] = order_items["Category_Diversity"].fillna(0)
    del items, products, enriched_items
    _log("Olist: aggregating payments and reviews")
    payments = pd.read_csv(paths["payments"], usecols=["order_id", "payment_value"])
    raw_rows += len(payments)
    order_payments = payments.groupby("order_id", as_index=False).agg(Payment_Value=("payment_value", "sum"))
    del payments
    reviews = pd.read_csv(paths["reviews"], usecols=["order_id", "review_score"])
    raw_rows += len(reviews)
    order_reviews = reviews.groupby("order_id", as_index=False).agg(Average_Review_Score=("review_score", "mean"))
    del reviews

    # Every joined source is first reduced to one row per order.
    _log("Olist: joining one-row-per-order aggregates")
    order_level = order_core.merge(order_items, on="order_id", how="inner", validate="one_to_one")
    del order_core, order_items
    order_level = order_level.merge(order_payments, on="order_id", how="left", validate="one_to_one")
    del order_payments
    order_level = order_level.merge(order_reviews, on="order_id", how="left", validate="one_to_one")
    del order_reviews
    order_level["Payment_Value"] = order_level["Payment_Value"].fillna(order_level["item_value"] + order_level["freight_value"])
    snapshot = order_level["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)

    _log("Olist: aggregating customer features")
    features = order_level.groupby("customer_unique_id").agg(
        First_Purchase=("order_purchase_timestamp", "min"),
        Last_Purchase=("order_purchase_timestamp", "max"),
        Frequency=("order_id", "nunique"),
        Monetary=("Payment_Value", "sum"),
        Unique_Products=("Unique_Products", "sum"),
        Category_Diversity=("Category_Diversity", "sum"),
        Average_Basket_Size=("Item_Count", "mean"),
        Average_Review_Score=("Average_Review_Score", "mean"),
    )
    order_level_count = len(order_level)
    del order_level
    gc.collect()
    features["Recency_Days"] = (snapshot - features["Last_Purchase"].dt.normalize()).dt.days
    features["Tenure_Days"] = (features["Last_Purchase"].dt.normalize() - features["First_Purchase"].dt.normalize()).dt.days
    features["Average_Order_Value"] = _safe_rate(features["Monetary"], features["Frequency"])
    features = features.drop(columns=["First_Purchase", "Last_Purchase"]).reset_index()
    quality = _quality_report(
        raw_rows, order_level_count, 0, 0,
        delivered_orders=delivered_count, excluded_non_delivered_orders=invalid_order_rows,
        order_level_rows=order_level_count, customer_count=len(features),
        repeat_customer_rate=round(float((features["Frequency"] > 1).mean()), 6),
        snapshot_date=str(snapshot.date()),
        rules="Only delivered orders with items are included. Cancelled, unavailable, and other statuses are excluded.",
    )
    return features, quality


def build_customer_features(dataset: str) -> tuple[pd.DataFrame, dict]:
    if dataset == "online_retail_ii":
        return build_online_retail_features()
    if dataset == "olist":
        return build_olist_features()
    raise ValueError(f"Unsupported dataset: {dataset}")
