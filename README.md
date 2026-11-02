# Comparative Evaluation of Data Normalization Techniques for Non-Gaussian Biomedical Signals

This Python project compares **Min-Max scaling**, **Quantile Transformation**, and **Box-Cox transformation** for both ECG-like and EEG-like datasets. It evaluates two classifiers (Random Forest and RBF-SVM) using accuracy, macro precision, macro recall, and macro F1-score, and saves visualizations and reports.

## Important scientific note

The default run uses **synthetic demonstration data** so you can verify the code immediately. Those results are not real ECG/EEG research findings and must not be presented as results from biomedical recordings. For the actual research, obtain suitable ECG and EEG datasets, document their licenses and preprocessing, and run this program on the real CSV files.

## 1. Open the project in VS Code

In Terminal:

```bash
cd ~/biomedical-normalization-research
code .
```

If your folder is somewhere else, `cd` to that folder instead.

## 2. Create and activate a virtual environment (Mac)

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the demo first

```bash
python main.py
```

This generates results for both synthetic ECG-like and EEG-like datasets. You should see a metrics table in Terminal and files inside `results/`.

## 4. CSV format for real datasets

Supply one row per observation/window/recording, numeric feature columns, and one label column named `label`. Example:

```csv
feature_1,feature_2,feature_3,label
0.12,0.45,0.22,normal
0.14,0.39,0.28,abnormal
```

Your features can be time samples or engineered features, but each row must have the same feature structure. Do not include identifiers, subject IDs, timestamps, or other leakage-prone columns as numeric features. If the target column has another name, use `--label-column target_name`.

Run with your files:

```bash
python main.py --ecg data/ecg.csv --eeg data/eeg.csv
```

If your CSV label field is called `class`:

```bash
python main.py --ecg data/ecg.csv --eeg data/eeg.csv --label-column class
```

The ECG and EEG CSVs must both use the same label-column name for this version. You can edit the command-line handling if they differ.

## 5. Outputs

The `results/` folder includes:

- `all_metrics.csv` — all metric values for each signal type, normalization, and classifier.
- `ecg_metric_comparison.png`, `eeg_metric_comparison.png` — macro-F1 comparison charts.
- `ecg_normalization_distributions.png`, `eeg_normalization_distributions.png` — distributions before and after transformations.
- `ecg_best_confusion_matrix.png`, `eeg_best_confusion_matrix.png` — confusion matrices for the best macro-F1 configuration.
- `ecg_best_classification_report.txt`, `eeg_best_classification_report.txt` — class-level precision, recall and F1.

## Methods

- **Min-Max:** maps each feature to a training-set range, usually [0, 1].
- **Quantile:** maps empirical feature quantiles to a normal distribution.
- **Box-Cox:** power transformation intended for strictly positive values. The implementation learns a positive shift from the training set and estimates a separate lambda for each feature.
- **Classifiers:** Random Forest and RBF-kernel SVM.
- **Metrics:** accuracy, macro-averaged precision, macro-averaged recall and macro-F1.

All transformations are fitted on the training split only to reduce train/test leakage. The current example uses a stratified random split. For biomedical data with repeated windows from the same person or recording, **use subject-wise or recording-wise splits** instead; random splitting can substantially overestimate performance.

## Research protocol recommendations

1. Choose real ECG and EEG datasets with clear labels and permission to use them.
2. Document subject counts, sampling frequency, class definitions, exclusions and preprocessing.
3. Split by participant/recording before creating overlapping windows.
4. Keep the test set untouched until final evaluation; use cross-validation on training data for tuning.
5. Repeat experiments across fixed random seeds or folds and report mean ± standard deviation.
6. Avoid claiming one normalization is universally best; results depend on dataset, features and model.
7. For raw ECG/EEG, consider signal-specific filtering/artifact handling and windowing before feature extraction.
8. Box-Cox changes data through a learned shift. Document that shift and compare it fairly with the other methods.

## Troubleshooting

- `code: command not found`: open VS Code, press `Cmd+Shift+P`, run **Shell Command: Install 'code' command in PATH**, then reopen Terminal.
- Missing library: activate `venv` and run `pip install -r requirements.txt`.
- Missing CSV: verify the path and filename.
- Label column error: pass the correct column with `--label-column`.
