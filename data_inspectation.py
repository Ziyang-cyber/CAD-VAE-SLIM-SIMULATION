from pathlib import Path
import numpy as np
import pandas as pd


#   0= REF/REF -> [0, 1]
#   1= REF/ALT -> [1, 1]
#   2= ALT/ALT -> [1, 0]
#   ./. = missing -> [0, 0]

root = Path("/home/zzhang/scratch/simulation_shuyu/data")

X_path = next(root.rglob("combined_neutral_and_qtl_dosage.npy"))

X = np.load(X_path, mmap_mode="r")

y = pd.read_csv(
    X_path.parents[1] / "individuals.tsv",
    sep=r"\\t|\t", engine="python"
)


print("File",X_path)
print("X",X.shape, "\n",X[:10,:100])
print("num of env labels",y["population"].unique())
print("y",y.shape,"\n",y.head(100).to_string(index=False))
