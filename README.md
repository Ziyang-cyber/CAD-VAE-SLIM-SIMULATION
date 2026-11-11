# CAD-VAE slim Simulation

Run one simulation repetition with a single train validation/test split.

## Environment

```bash
conda env create -f environment.yml
conda activate vae
```

## Scripts

| Script | Purpose |
| --- | --- |
| `data_insepctation.sh` | Check genotype data and labels. |
| `run_simulation_cmi.sh` | Train with different CMI weights and selected CLS values. |
| `run_simulation_lri.sh` | Train with different LRI weights. |
| `run_simulation_tc.sh` | Train with different TC weights. |
| `run_plt_cmi.sh` | Plot AUC versus CMI, one figure per CLS. |
| `run_plt_lri.sh` | Plot AUC versus LRI. |
| `run_plt_tc.sh` | Plot AUC versus TC. |

## Run

Update the paths, loss weights, and Slurm resources in the scripts first.
Run commands from the `code` directory.


1. Create the log directory and check the data:

   ```bash
   mkdir -p log
   sbatch data_insepctation.sh
   ```

2. After checking the data, submit the training experiment you need:

   ```bash
   sbatch run_simulation_cmi.sh
   sbatch run_simulation_lri.sh
   sbatch run_simulation_tc.sh
   ```

3. After its training jobs finish, submit the matching plotting script:

   ```bash
   sbatch run_plt_cmi.sh
   sbatch run_plt_lri.sh
   sbatch run_plt_tc.sh
   ```

Training reads genotypes from `neutral_snps/combined_neutral_and_qtl_dosage.npy`
and labels from `individuals.tsv`: `phenotype_binary` is Y, and `population` is S.

Results are saved to the configured results directory, AUC figures are saved under `auc_plots/`.
