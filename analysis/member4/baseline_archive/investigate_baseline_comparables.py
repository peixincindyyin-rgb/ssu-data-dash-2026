"""Optional reproduction of the three largest baseline errors using full listings.

Download listings.parquet from the official dataset, then run:
  python investigate_baseline_comparables.py --listings path/to/listings.parquet
This helper is specifically for the official 10-nearest-sales baseline.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listings",type=Path,required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    x = pd.read_csv(root/"tables/row_errors.csv")
    md = pd.read_parquet(a.listings) if a.listings.suffix == ".parquet" else pd.read_csv(a.listings)
    md["date_sold"] = pd.to_datetime(md.date_sold)
    train = md.sort_values(["date_sold","ml_num"],ascending=[False,True])
    rows=[]
    for h in x[x.track=="off_market"].nlargest(3,"ape_pct").itertuples():
        cutoff=min(pd.Timestamp(h.month+"-01"),pd.Timestamp(h.date_listed))
        t=train[(train.house_type_name==h.house_type_name)&(train.date_sold<cutoff)]
        source=md[md.ml_num==h.ml_num].iloc[0]
        d=np.sqrt(((t.longitude.to_numpy()-source.longitude)*.72)**2+(t.latitude.to_numpy()-source.latitude)**2)
        comp=t.iloc[np.argsort(d,kind="stable")[:10]]
        prediction=float(np.round(np.median(comp.sold_price)))
        if prediction != h.predicted_price:
            raise ValueError("Predictions do not match the official nearest-sales baseline")
        for rank,c in enumerate(comp.itertuples(),1):
            rows.append({"target_id":h.ml_num,"target_area":h.house_area,
                         "target_sold_price":h.sold_price,"target_prediction":h.predicted_price,
                         "target_listed":h.date_listed,"rank":rank,"comparable_id":c.ml_num,
                         "comparable_community":c.community,"comparable_area":c.house_area,
                         "comparable_sold_price":c.sold_price,"comparable_sale_date":c.date_sold})
    pd.DataFrame(rows).to_csv(root/"tables/extreme_case_comparables.csv",index=False)
    print("Reproduced three extreme-case baseline predictions and their 30 comparables.")

if __name__ == "__main__":
    main()
