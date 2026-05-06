import pandas as pd

df = pd.read_parquet("featuresdf.pq")

print("Before: ",df.shape)
dfsmall = df[ df.index %100==0] 

print("After: ",dfsmall.shape)

dfsmall.to_parquet("featuresdf_small.pq")
