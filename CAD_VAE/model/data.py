from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    rep_dir: str
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 1024


class GenotypeDataset(Dataset):
    """Return (x, y, s) for one individual."""

    def __init__(self, genotype, y, s, indices):
        self.genotype = genotype
        self.y = y
        self.s = s
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        g = self.genotype[row]

        # Use the same two-channel encoding as cricket:
        # 0 -> [0, 1]
        # 1 -> [1, 1]
        # 2 -> [1, 0]
        x = np.stack((g > 0, g < 2), axis=-1)
        x = x.astype(np.float32).reshape(-1)

        return torch.from_numpy(x), self.y[row], self.s[row]


class SimulationDataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        cfg = self.cfg
        rep_dir = Path(cfg.rep_dir)

        # 1. Load the genotype matrix and metadata.
        self.genotype = np.load(
            rep_dir / "neutral_snps/combined_neutral_and_qtl_dosage.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        self.metadata_all = pd.read_csv(
            rep_dir / "individuals.tsv",
            sep=r"\\t|\t",
            engine="python",
            dtype={"individual_id": str},
        )

        if self.genotype.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")

        self.n_samples, self.n_snps = self.genotype.shape
        self.input_dim = self.n_snps * 2

        if len(self.metadata_all) != self.n_samples:
            raise ValueError("X and individuals.tsv have different row counts.")

        columns = ["individual_id", "phenotype_binary", "population"]
        if self.metadata_all[columns].isna().any().any():
            raise ValueError("Missing sample IDs, phenotype labels, or populations.")

        # Validate genotypes in chunks to limit temporary memory usage.
        for start in range(0, self.n_samples, 128):
            if not np.isin(self.genotype[start:start + 128], [0, 1, 2]).all():
                raise ValueError("X must contain only 0, 1, and 2 without missing values.")

        # 2. Keep sample IDs in the original row order.
        # Metadata row i must correspond to genotype row i.
        self.sample_ids_all = (
            self.metadata_all["individual_id"]
            .str.strip()
            .to_numpy(dtype=str)
        )

        if (
            (self.sample_ids_all == "").any()
            or len(np.unique(self.sample_ids_all)) != self.n_samples
        ):
            raise ValueError("Sample IDs must be nonempty and unique.")

        # 3. Use the binary phenotype as Y.
        y = self.metadata_all["phenotype_binary"].to_numpy()

        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("phenotype_binary must contain both classes 0 and 1.")

        self.y_all = y.astype(np.int64)
        self.num_y_classes = 2

        # 4. Encode populations as S labels: 0, ..., K-1.
        populations, s = np.unique(
            self.metadata_all["population"].to_numpy(),
            return_inverse=True,
        )

        self.s_all = s.astype(np.int64)
        self.num_s_classes = len(populations)

        if self.num_s_classes < 2:
            raise ValueError("At least two population classes are required.")

        self.class_mapping = pd.DataFrame({
            "s_index": np.arange(self.num_s_classes),
            "population": populations,
        })

        # 5. Jointly stratify by (Y, S) for a single 70/10/20 split.
        indices = np.arange(self.n_samples)
        strata = self.y_all * self.num_s_classes + self.s_all

        train_val, test = train_test_split(
            indices,
            test_size=0.20,
            stratify=strata,
            random_state=cfg.seed,
        )

        train, val = train_test_split(
            train_val,
            test_size=0.125,  # 12.5% of 80% gives 10% of all samples.
            stratify=strata[train_val],
            random_state=cfg.seed,
        )

        self.split_indices = {
            "train": train,
            "val": val,
            "test": test,
        }

        self.split_info = {
            "rep_dir": str(rep_dir),
            "n_samples": self.n_samples,
            "n_snps": self.n_snps,
            "input_dim": self.input_dim,
            "num_y_classes": self.num_y_classes,
            "num_s_classes": self.num_s_classes,
            "split_sizes": {
                name: len(idx)
                for name, idx in self.split_indices.items()
            },
        }

    def dataloader(self, split_name, shuffle=False):
        dataset = GenotypeDataset(
            self.genotype,
            self.y_all,
            self.s_all,
            self.split_indices[split_name],
        )

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            drop_last=False,
        )

    def train_dataloader(self):
        return self.dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self.dataloader("val")

    def test_dataloader(self):
        return self.dataloader("test")