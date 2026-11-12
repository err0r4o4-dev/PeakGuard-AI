from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, fbeta_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 312
SOURCE_URL = 'https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption'


def load_data(mode='local', csv_path='data/PJME_hourly.csv', drive_path=None):
    if mode == 'upload':
        try:
            from google.colab import files
        except ImportError as error:
            raise RuntimeError('Upload mode requires Colab. Outside Colab use local mode.') from error
        uploaded = files.upload()
        matches = [name for name in uploaded if Path(name).name == 'PJME_hourly.csv']
        if len(matches) != 1:
            raise ValueError('Upload exactly one file named PJME_hourly.csv.')
        csv_path = matches[0]
    elif mode == 'drive':
        if not drive_path:
            raise ValueError('Set DRIVE_PATH to the full PJME_hourly.csv path.')
        try:
            from google.colab import drive
        except ImportError as error:
            raise RuntimeError('Drive mounting requires Colab. Use local mode elsewhere.') from error
        drive.mount('/content/drive')
        csv_path = drive_path
    elif mode == 'kaggle':
        import os
        import shutil
        import tempfile
        import zipfile
        if not shutil.which('kaggle'):
            raise RuntimeError('Install the optional kaggle package before using this mode.')
        configured = bool(os.environ.get('KAGGLE_API_TOKEN')) or bool(os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))
        config_dir = Path(os.environ.get('KAGGLE_CONFIG_DIR', Path.home() / '.kaggle'))
        if not configured and not (config_dir / 'kaggle.json').is_file():
            raise RuntimeError('Configure Kaggle credentials privately; never paste them into notebook cells.')
        folder = Path(csv_path).parent
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='peakguard-kaggle-') as download_dir:
            result = subprocess.run(['kaggle', 'datasets', 'download', '-d', 'robikscube/hourly-energy-consumption', '-f', 'PJME_hourly.csv', '-p', download_dir], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Kaggle download failed. Check account access/credentials privately or use upload; command output suppressed to protect credentials.')
            downloaded = Path(download_dir) / 'PJME_hourly.csv'
            if downloaded.is_file():
                shutil.copyfile(downloaded, csv_path)
            else:
                archive = Path(download_dir) / 'PJME_hourly.csv.zip'
                if not archive.is_file():
                    raise FileNotFoundError('Kaggle did not produce PJME_hourly.csv or PJME_hourly.csv.zip.')
                with zipfile.ZipFile(archive) as bundle:
                    info = bundle.getinfo('PJME_hourly.csv')
                    if info.file_size > 100_000_000:
                        raise ValueError('Unexpectedly large CSV archive.')
                    Path(csv_path).write_bytes(bundle.read(info))
    elif mode != 'local':
        raise ValueError('MODE must be local, upload, drive or kaggle.')
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}. Upload/download PJME_hourly.csv or set the correct path.')
    raw = pd.read_csv(path)
    provenance = {'source_url': SOURCE_URL, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'raw_rows': len(raw), 'columns': list(raw.columns)}
    return raw, provenance


def clean_data(raw):
    required = {'Datetime', 'PJME_MW'}
    if not required.issubset(raw.columns):
        raise ValueError('Required columns: Datetime, PJME_MW.')
    timestamps = pd.to_datetime(raw['Datetime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    if getattr(timestamps.dt, 'tz', None) is not None:
        raise ValueError('Expected naive source timestamps; define timezone policy before conversion.')
    values = pd.to_numeric(raw['PJME_MW'], errors='coerce')
    invalid_values = values.isna() | ~np.isfinite(values) | (values <= 0)
    frame = pd.DataFrame({'time': timestamps, 'demand_mw': values.mask(invalid_values)})
    report = {'raw_rows': len(raw), 'raw_missing_by_column': raw[list(sorted(required))].isna().sum().to_dict(), 'raw_dtypes': raw[list(sorted(required))].dtypes.astype(str).to_dict(), 'invalid_datetime_rows': int(timestamps.isna().sum()), 'invalid_demand_rows': int(invalid_values.sum()), 'duplicate_time_rows': int(timestamps.dropna().duplicated(keep=False).sum()), 'input_sorted': bool(timestamps.dropna().is_monotonic_increasing)}
    frame = frame.dropna(subset=['time']).sort_values('time')
    if frame.empty:
        raise ValueError('No valid timestamps remain.')
    if (frame['time'] != frame['time'].dt.floor('h')).any():
        raise ValueError('Non-hour-aligned timestamps found; resolve explicitly before training.')
    duration = frame['time'].max() - frame['time'].min()
    if duration.total_seconds() > 366 * 50 * 86400:
        raise ValueError('Date span exceeds the 50-year safety bound; inspect datetime input.')
    groups = frame.groupby('time')['demand_mw']
    conflicts = groups.nunique(dropna=False) > 1
    hourly = groups.first()
    hourly.loc[conflicts] = np.nan
    index = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h', name='issue_time')
    cleaned = hourly.reindex(index).to_frame('demand_mw')
    report.update({'conflicting_duplicate_hours': int(conflicts.sum()), 'collapsed_duplicate_rows': int(len(frame) - len(hourly)), 'missing_grid_hours': int(len(index) - len(hourly)), 'grid_rows': len(cleaned), 'observed_clean_rows': int(cleaned['demand_mw'].notna().sum()), 'missing_after_cleaning': int(cleaned['demand_mw'].isna().sum()), 'start': str(index.min()), 'end': str(index.max()), 'observed_interval_counts_hours': hourly.index.to_series().diff().dt.total_seconds().div(3600).value_counts().sort_index().to_dict()})
    return cleaned, report


def make_features(cleaned):
    demand = cleaned['demand_mw']
    target_time = cleaned.index + np.timedelta64(1, "h")
    frame = pd.DataFrame(index=cleaned.index)
    frame['current_mw'] = demand
    for hours in (1, 24, 168):
        frame[f'lag_{hours}h_mw'] = demand.shift(hours)
    for hours in (24, 168):
        window = demand.rolling(hours, min_periods=hours)
        frame[f'rolling_mean_{hours}h_mw'] = window.mean()
        frame[f'rolling_std_{hours}h_mw'] = window.std()
    frame['trend_1h_mw'] = demand - demand.shift(1)
    frame['target_hour'] = target_time.hour
    frame['target_weekday'] = target_time.dayofweek
    frame['target_month'] = target_time.month
    frame['target_weekend'] = (target_time.dayofweek >= 5).astype(int)
    feature_columns = list(frame.columns)
    frame['day_ago_target_mw'] = demand.shift(23)
    frame['target_time'] = target_time
    frame['target_mw'] = demand.shift(-1)
    return frame, feature_columns


def split_data(cleaned, frame, feature_columns):
    total = len(cleaned)
    first, second = int(total * .70), int(total * .85)
    if min(first, second - first, total - second) < 200:
        raise ValueError('Insufficient hourly history: each raw partition needs at least 200 hours; use about 1000+ rows.')
    threshold = float(cleaned['demand_mw'].iloc[:first].quantile(.90))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Training P90 is unavailable.')
    boundaries = [cleaned.index[0], cleaned.index[first], cleaned.index[second], cleaned.index[-1] + np.timedelta64(1, "h")]
    partitions, audit = {}, {}
    for name, lower, upper in zip(('train', 'validation', 'test'), boundaries[:-1], boundaries[1:]):
        eligible = frame.loc[(frame.index >= lower) & (frame.index < upper) & (frame['target_time'] < upper)]
        selected = eligible.dropna(subset=feature_columns + ['target_mw', 'day_ago_target_mw']).copy()
        if selected.empty:
            raise ValueError(f'No complete samples in {name}. Inspect missing hours or provide more history.')
        selected['is_peak'] = (selected['target_mw'] >= threshold).astype(int)
        partitions[name] = selected
        audit[name] = {'raw_start': str(lower), 'raw_end_exclusive': str(upper), 'candidate_rows_after_boundary_purge': len(eligible), 'usable_rows': len(selected), 'dropped_incomplete_rows': len(eligible) - len(selected), 'normal_samples': int((selected['is_peak'] == 0).sum()), 'peak_samples': int(selected['is_peak'].sum()), 'peak_fraction': float(selected['is_peak'].mean()), 'normal_to_peak_ratio': float((selected['is_peak'] == 0).sum() / selected['is_peak'].sum()) if selected['is_peak'].sum() else None}
    assert partitions['train']['target_time'].max() < partitions['validation'].index.min()
    assert partitions['validation']['target_time'].max() < partitions['test'].index.min()
    return partitions, threshold, audit


def metric_row(name, actual, predicted, scores):
    actual = np.asarray(actual)
    return {'model': name, 'accuracy': float(accuracy_score(actual, predicted)), 'precision': float(precision_score(actual, predicted, zero_division=0)), 'recall': float(recall_score(actual, predicted, zero_division=0)), 'f1': float(fbeta_score(actual, predicted, beta=1, zero_division=0)), 'f2': float(fbeta_score(actual, predicted, beta=2, zero_division=0)), 'average_precision': float(average_precision_score(actual, scores)) if len(np.unique(actual)) == 2 else None}


def choose_cutoff(actual, probabilities):
    candidates = np.round(np.linspace(.05, .95, 37), 3)
    rows = [(metric_row('candidate', actual, probabilities >= cutoff, probabilities), float(cutoff)) for cutoff in candidates]
    return max(rows, key=lambda item: (item[0]['f2'], item[0]['recall'], item[0]['precision'], item[1]))[1]


def train_candidates(partitions, feature_columns, threshold):
    training, validation = partitions['train'], partitions['validation']
    for name, subset in (('train', training), ('validation', validation)):
        if subset['is_peak'].nunique() < 2:
            raise ValueError(f'{name} has one class. Do not tune using test; reconsider history or document this dataset limitation.')
    models = {'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED)), 'Random Forest': RandomForestClassifier(n_estimators=160, max_depth=16, min_samples_leaf=5, class_weight='balanced_subsample', random_state=SEED, n_jobs=2)}
    candidates = {'Persistence': {'kind': 'baseline', 'column': 'current_mw', 'cutoff': threshold}, 'Previous Day': {'kind': 'baseline', 'column': 'day_ago_target_mw', 'cutoff': threshold}}
    for name, model in models.items():
        model.fit(training[feature_columns], training['is_peak'])
        scores = model.predict_proba(validation[feature_columns])[:, 1]
        candidates[name] = {'kind': 'model', 'estimator': model, 'cutoff': choose_cutoff(validation['is_peak'], scores)}
    rows = []
    for name, candidate in candidates.items():
        scores, predicted = predict_candidate(candidate, validation, feature_columns)
        rows.append(metric_row(name, validation['is_peak'], predicted, scores) | {'decision_cutoff': candidate['cutoff']})
    table = pd.DataFrame(rows).sort_values(['f2', 'recall', 'precision', 'model'], ascending=[False, False, False, True], kind='stable').reset_index(drop=True)
    winner = str(table.iloc[0]['model'])
    return candidates, table, winner


def predict_candidate(candidate, frame, feature_columns):
    if candidate['kind'] == 'baseline':
        scores = frame[candidate['column']].to_numpy()
    else:
        scores = candidate['estimator'].predict_proba(frame[feature_columns])[:, 1]
    return scores, (scores >= candidate['cutoff']).astype(int)


def evaluate_test(candidates, test, feature_columns, selected_name):
    rows, predictions = [], pd.DataFrame(index=test.index)
    predictions['target_time'] = test['target_time']
    predictions['actual_mw'] = test['target_mw']
    predictions['actual_peak'] = test['is_peak']
    for name, candidate in candidates.items():
        scores, predicted = predict_candidate(candidate, test, feature_columns)
        rows.append(metric_row(name, test['is_peak'], predicted, scores) | {'selected_on_validation': name == selected_name, 'decision_cutoff': candidate['cutoff']})
        predictions[f'{name}_score'] = scores
        predictions[f'{name}_prediction'] = predicted
    predictions['selected_prediction'] = predictions[f'{selected_name}_prediction']
    actual, predicted = predictions['actual_peak'], predictions['selected_prediction']
    predictions['error_type'] = np.select([(actual == 1) & (predicted == 0), (actual == 0) & (predicted == 1)], ['FN', 'FP'], default='Correct')
    return pd.DataFrame(rows), predictions


def save_figure(figure, name, output_dir):
    destination = Path(output_dir) / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(figure)


def exploratory_plots(cleaned, training, feature_columns, threshold, training_end, output_dir):
    sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
    history = cleaned.loc[cleaned.index < pd.Timestamp(training_end), 'demand_mw'].dropna()
    captions = {}
    figure, axis = plt.subplots(figsize=(11, 4))
    history.resample('D').mean().plot(ax=axis, linewidth=.8)
    axis.axhline(threshold, color='#C74E31', linestyle='--', label='Hourly training P90')
    axis.set(title='Training demand over time (daily means)', xlabel='Source date (naive local label)', ylabel='Demand (MW)')
    axis.legend()
    save_figure(figure, 'eda_demand_over_time', output_dir)
    captions['eda_demand_over_time'] = f'ข้อมูล Train มีค่าเฉลี่ย {history.mean():,.1f} MW; เส้น P90 ของข้อมูลรายชั่วโมงเท่ากับ {threshold:,.1f} MW ไม่ใช่ P90 ของค่าเฉลี่ยรายวัน'
    for key, grouping, title, xlabel in [('hour', history.index.hour, 'Average training demand by hour', 'Source hour (0-23)'), ('weekday', history.index.dayofweek, 'Average training demand by weekday', 'Weekday (0=Monday)'), ('month', history.index.month, 'Average training demand by month', 'Month (1-12)')]:
        summary = history.groupby(grouping).mean()
        figure, axis = plt.subplots(figsize=(8, 4))
        summary.plot.bar(ax=axis, color='#217A7B')
        axis.set(title=title, xlabel=xlabel, ylabel='Mean demand (MW)')
        save_figure(figure, f'eda_{key}', output_dir)
        captions[f'eda_{key}'] = f'กลุ่ม {key} ที่มีค่าเฉลี่ย Train สูงสุดคือ {summary.idxmax()} ({summary.max():,.1f} MW); เป็นความสัมพันธ์ ไม่ใช่หลักฐานเชิงสาเหตุ'
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.histplot(history, bins=60, ax=axis, color='#217A7B')
    axis.axvline(threshold, color='#C74E31', linestyle='--')
    axis.set(title='Training demand distribution', xlabel='Demand (MW)', ylabel='Observed hours (count)')
    save_figure(figure, 'eda_distribution', output_dir)
    captions['eda_distribution'] = f'นิยาม Peak คือ demand >= Train P90; อัตรา Peak ในชั่วโมงที่สังเกตได้ของ Train = {(history >= threshold).mean():.2%}'
    peak_by_hour = (history >= threshold).groupby(history.index.hour).mean() * 100
    figure, axis = plt.subplots(figsize=(8, 4))
    peak_by_hour.plot.bar(ax=axis, color='#C74E31')
    axis.set(title='Training peak frequency by source hour', xlabel='Source hour (0-23)', ylabel='Peak hours (%)')
    save_figure(figure, 'eda_peak_distribution', output_dir)
    captions['eda_peak_distribution'] = f'สัดส่วน Peak สูงสุดใน Train อยู่ที่ชั่วโมง {peak_by_hour.idxmax()} ({peak_by_hour.max():.2f}%)'
    heat = pd.DataFrame({'hour': history.index.hour, 'weekday': history.index.dayofweek, 'demand': history.to_numpy()}).pivot_table(index='weekday', columns='hour', values='demand', aggfunc='mean')
    figure, axis = plt.subplots(figsize=(11, 4))
    sns.heatmap(heat, cmap='YlGnBu', ax=axis, cbar_kws={'label': 'Mean demand (MW)'})
    axis.set(title='Training hourly demand heatmap', xlabel='Source hour (0-23)', ylabel='Weekday (0=Monday)')
    save_figure(figure, 'eda_heatmap', output_dir)
    captions['eda_heatmap'] = 'สีแสดงค่าเฉลี่ย MW ของ Train ตามวันและชั่วโมง ใช้ค้นหารูปแบบ ไม่ใช้ Test ออกแบบ Features'
    figure, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(training[feature_columns].corr(), cmap='vlag', center=0, vmin=-1, vmax=1, ax=axis, cbar_kws={'label': 'Pearson correlation (unitless)'})
    axis.set(title='Training feature correlation', xlabel='Input feature (MW or calendar unit)', ylabel='Input feature (MW or calendar unit)')
    save_figure(figure, 'eda_correlation', output_dir)
    captions['eda_correlation'] = 'Correlation คำนวณจาก Features ของ Train เท่านั้น; ค่าสูงไม่ได้ยืนยันเหตุและผล และ lags มีความสัมพันธ์กันได้'
    return captions


def evaluation_plots(candidates, test, feature_columns, metrics, predictions, selected_name, output_dir):
    figure, axis = plt.subplots(figsize=(10, 5))
    metrics.set_index('model')[['precision', 'recall', 'f2', 'average_precision']].plot.bar(ax=axis)
    axis.set(title='Frozen models: final test comparison', xlabel='Model (winner fixed on validation)', ylabel='Metric (0-1, unitless)', ylim=(0, 1.05))
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Metrics")
    axis.tick_params(axis="x", labelrotation=20)
    save_figure(figure, 'model_comparison', output_dir)
    figure, axis = plt.subplots(figsize=(5, 4))
    matrix = confusion_matrix(predictions['actual_peak'], predictions['selected_prediction'], labels=[0, 1])
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axis, xticklabels=['Normal', 'Peak'], yticklabels=['Normal', 'Peak'], cbar_kws={'label': 'Samples (count)'})
    axis.set(title=f'Test confusion matrix: {selected_name}', xlabel='Predicted class', ylabel='Actual class')
    save_figure(figure, 'confusion_matrix', output_dir)
    if test['is_peak'].nunique() == 2:
        figure, axis = plt.subplots(figsize=(8, 5))
        for name, candidate in candidates.items():
            scores, _ = predict_candidate(candidate, test, feature_columns)
            precision, recall, _ = precision_recall_curve(test['is_peak'], scores)
            axis.plot(recall, precision, label=name)
        axis.axhline(test['is_peak'].mean(), color='gray', linestyle='--', label='Test prevalence')
        axis.set(title='Final test precision-recall curves', xlabel='Recall (unitless)', ylabel='Precision (unitless)', xlim=(0, 1), ylim=(0, 1.02))
        axis.legend()
        save_figure(figure, 'precision_recall', output_dir)
    importance = pd.Series(candidates['Random Forest']['estimator'].feature_importances_, index=feature_columns).sort_values()
    figure, axis = plt.subplots(figsize=(9, 6))
    importance.plot.barh(ax=axis, color='#217A7B')
    axis.set(title='Random Forest training impurity importance', xlabel='Impurity importance (unitless)', ylabel='Input feature')
    save_figure(figure, 'feature_importance', output_dir)
    importance.rename('importance').to_csv(Path(output_dir) / 'feature_importance.csv')
    tables = {}
    for name, group in [('hour', predictions['target_time'].dt.hour), ('month', predictions['target_time'].dt.month)]:
        temporary = predictions.assign(group=group)
        rows = []
        for label, subset in temporary.groupby('group'):
            actual = subset['actual_peak']
            predicted = subset['selected_prediction']
            rows.append({'group': int(label), 'samples': len(subset), 'actual_peaks': int(actual.sum()), 'FP': int((subset['error_type'] == 'FP').sum()), 'FN': int((subset['error_type'] == 'FN').sum()), 'recall': float(recall_score(actual, predicted, zero_division=0)) if actual.sum() else None})
        table = pd.DataFrame(rows)
        table.to_csv(Path(output_dir) / f'errors_by_{name}.csv', index=False)
        tables[name] = table
        figure, axis = plt.subplots(figsize=(9, 4))
        table.set_index('group')[['FP', 'FN']].plot.bar(ax=axis, color=['#217A7B', '#C74E31'])
        axis.set(title=f'Selected model test errors by target {name}', xlabel=f'Target {name} (source labels)', ylabel='Incorrect predictions (count)')
        save_figure(figure, f'errors_by_{name}', output_dir)
    return tables


def simulate_alert(cleaned, artifact):
    frame, columns = make_features(cleaned)
    if columns != artifact['feature_columns']:
        raise ValueError('Feature schema differs from the trained artifact.')
    latest = frame.iloc[[-1]]
    if latest[columns + ['day_ago_target_mw']].isna().any().any():
        raise ValueError('Latest hour lacks complete causal history; do not issue a prediction.')
    scores, predicted = predict_candidate(artifact['candidate'], latest, columns)
    return {'simulation_only': True, 'issue_time': str(latest.index[0]), 'target_time': str(latest['target_time'].iloc[0]), 'model': artifact['selected_name'], 'score': float(scores[0]), 'score_unit': 'MW' if artifact['candidate']['kind'] == 'baseline' else 'uncalibrated model probability', 'decision_cutoff': artifact['candidate']['cutoff'], 'alert': bool(predicted[0]), 'actual_future_known': False}


def export_results(output_dir, provenance, quality, audit, threshold, captions, validation_metrics, test_metrics, predictions, selected_name, candidates, feature_columns):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_metrics.to_csv(output / 'validation_metrics.csv', index=False)
    test_metrics.to_csv(output / 'test_metrics.csv', index=False)
    predictions.to_csv(output / 'test_predictions.csv', index_label='issue_time')
    predictions.loc[predictions['error_type'] != 'Correct'].to_csv(output / 'error_cases.csv', index_label='issue_time')
    report = classification_report(predictions['actual_peak'], predictions['selected_prediction'], labels=[0, 1], target_names=['Normal', 'Peak'], output_dict=True, zero_division=0)
    artifact = {'candidate': candidates[selected_name], 'selected_name': selected_name, 'feature_columns': feature_columns, 'peak_threshold_mw': threshold, 'source_time_policy': 'naive source labels; unresolved DST ambiguity; complete observation t available', 'seed': SEED}
    joblib.dump(artifact, output / 'peakguard_model.joblib')
    packages = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'joblib')}
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'packages': packages, 'seed': SEED, 'provenance': provenance, 'quality': quality, 'split_audit': audit, 'peak_threshold_mw': threshold, 'selected_on_validation': selected_name, 'validation_metrics': validation_metrics.to_dict('records'), 'test_metrics': test_metrics.to_dict('records'), 'classification_report': report, 'eda_captions_th': captions, 'limitations': ['retrospective simulation only', 'naive timestamps: exact elapsed time across DST not established', 'one temporal split; no causal/financial savings claim', 'teacher PDF not supplied', 'AI team decisions pending actual team confirmation']}
    (output / 'run_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'figure_captions_th.json').write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# PeakGuard AI — A2 Poster Summary', '', '**ชื่อไทย:** ระบบพยากรณ์และแจ้งเตือนช่วงเวลาที่มีความต้องการใช้ไฟฟ้าสูงด้วยการเรียนรู้ของเครื่อง', '**English:** Peak Electricity Demand Prediction and Early Warning Using Machine Learning', '**SDG:** Goal 7 — Affordable and Clean Energy', '', '## R1 — Problem Statement', 'จำลองการเตือนล่วงหน้า 1 ชั่วโมงให้ผู้วางแผนจัดการความต้องการไฟฟ้าเตรียมมาตรการลดโหลดที่เลื่อนได้ โดยยังไม่มีการเชื่อมต่อระบบจริงหรือตรวจวัดผลประหยัด', '', '## R2 — Dataset & Preparation', f'PJM / PJME_hourly.csv: {quality["raw_rows"]:,} แถว; ช่วง {quality["start"]} ถึง {quality["end"]}; หน่วย MW (กำลังไฟฟ้า ไม่ใช่ MWh)', f'แบ่งเวลา 70/15/15 ก่อนคัดตัวอย่าง; Train P90 = {threshold:,.2f} MW; ลบตัวอย่างที่ Target ข้ามชุดข้อมูล', f'ชั่วโมงที่หายจาก grid {quality["missing_grid_hours"]}; duplicate ขัดแย้ง {quality["conflicting_duplicate_hours"]}; ไม่เติมจากอนาคต', '', '## R3 — EDA & Insights']
    lines += [f'- {text}' for text in captions.values()]
    lines += ['', '## R4 — Baseline vs Machine Learning', f'เลือก **{selected_name}** จาก Validation F2 แล้วล็อกก่อนเปิด Test; ตารางต่อไปนี้เป็นผล Test ขั้นสุดท้าย ไม่ใช้เลือกโมเดล', '', '| Model | Accuracy | Precision | Recall | F1 | F2 | AP (PR-AUC summary) |', '|---|---:|---:|---:|---:|---:|---:|']
    for record in test_metrics.to_dict('records'):
        numbers = [f'{record[key]:.4f}' if pd.notna(record[key]) else 'Undefined (one class)' for key in ('accuracy', 'precision', 'recall', 'f1', 'f2', 'average_precision')]
        lines.append('| ' + record['model'] + ' | ' + ' | '.join(numbers) + ' |')
    false_positive = int((predictions['error_type'] == 'FP').sum())
    false_negative = int((predictions['error_type'] == 'FN').sum())
    winner_f2 = float(test_metrics.loc[test_metrics['model'] == selected_name, 'f2'].iloc[0])
    baseline_f2 = float(test_metrics.loc[test_metrics['model'] == 'Persistence', 'f2'].iloc[0])
    lines += ['', f'F2 ต่างจาก Persistence = {winner_f2 - baseline_f2:+.4f}; ไม่รับรองว่าจะชนะ Baseline ในทุกช่วงเวลา', '', '## R5 — Error Analysis', f'โมเดลที่เลือก: FP {false_positive:,} ตัวอย่าง, FN {false_negative:,} ตัวอย่าง; รายละเอียด outputs/error_cases.csv และ errors_by_hour/month.csv', 'FP อาจทำให้เตรียมมาตรการเกินจำเป็น; FN อาจทำให้ไม่เตรียมรับ Peak จึงใช้ Recall และ F2 เป็นหลัก ต้นทุนเงินจริงยังไม่ได้วัด', '', '## R6 — Recommendations & Limitations', 'ใช้เป็นข้อมูลประกอบการวางแผนและให้คนอนุมัติมาตรการก่อนลงมือ ต้องทดสอบ telemetry latency, timezone/DST, drift, calibration และผลทางธุรกิจก่อนใช้งานจริง', 'P90 เป็นนิยามเชิงสถิติ ไม่ใช่เกณฑ์วิศวกรรมของโครงข่าย และชุดข้อมูลสหรัฐฯ ไม่ยืนยันผลสำหรับประเทศไทย', '', '## R7 — AI Usage Log', 'ใช้ AI ช่วยออกแบบ causal pipeline, เขียนโค้ดและตรวจผล; การตรวจรับและกรณีที่ทีมแก้ไข/ปฏิเสธคำแนะนำ: Pending Team Confirmation ห้ามแต่งประวัติ', '', '## Recommended 300-DPI figures', '- outputs/figures/eda_heatmap.png', '- outputs/figures/model_comparison.png', '- outputs/figures/confusion_matrix.png', '- outputs/figures/precision_recall.png', '- outputs/figures/errors_by_hour.png', '', '## Sources and License', SOURCE_URL, 'Kaggle public metadata checked 2026-11-12: licenseName = CC0: Public Domain. This is the distributor metadata, not independent legal verification of upstream PJM rights.', 'Data dictionary verified from actual CSV: Datetime = naive source timestamp label; PJME_MW = PJME demand in MW per source description. Timezone convention not established.', f'CSV SHA-256: {provenance["sha256"]}', '', '## Team Must Complete', '- ชื่อสมาชิก รหัสนิสิต อาจารย์และสถาบัน', '- ตรวจข้อกำหนดเทียบ DS Capstone Project Requirements 1.pdf เมื่อได้รับไฟล์', '- บันทึกการตัดสินใจ AI จริงอย่างน้อย 1 กรณีที่แก้ไขหรือปฏิเสธ พร้อมเหตุผล', '- ตรวจบน Google Colab จริงและจัดวางโปสเตอร์ขนาด A2; ไฟล์นี้เป็นเนื้อหา ไม่ใช่โปสเตอร์สำเร็จ', '- Independent review: see validation/INDEPENDENT_REVIEW.md for scope, findings and status']
    (output.parent / 'POSTER_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result, artifact
