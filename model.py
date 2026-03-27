import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier   
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc as auc_score, precision_recall_curve
import seaborn as sns
import warnings
from scipy import stats
import os
import shap
import pickle
from copy import deepcopy

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ============================================================================
# 工具函数
# ============================================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = np.vstack([predictions_one, predictions_two])[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    p_value = 10 ** calc_pvalue(aucs, delongcov)[0][0]
    return p_value, aucs[0], aucs[1]

def find_optimal_threshold(y_true, y_probs, method='youden'):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    if method == 'youden':
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
    else:
        raise ValueError(f"未知method: {method}")
    return thresholds[optimal_idx], youden_index[optimal_idx], optimal_idx

def calculate_net_benefit(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    n = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

def get_expected_value(explainer):
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.atleast_1d(ev)
        return float(ev[1]) if len(ev) >= 2 else float(ev[0])
    return float(ev)

def bootstrap_metrics(y_true, y_probs, threshold=0.5, n_bootstrap=1000, random_state=42):
    """Bootstrap 95% CI for all metrics."""
    rng = np.random.RandomState(random_state)
    y_true = np.array(y_true); y_probs = np.array(y_probs)
    n = len(y_true)
    boot = {k: [] for k in ['AUC','Sensitivity','Specificity','Accuracy','F1','PPV','NPV']}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]; yp = y_probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot['AUC'].append(roc_auc_score(yt, yp))
        except Exception:
            continue
        ypred = (yp >= threshold).astype(int)
        cm = confusion_matrix(yt, ypred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        acc  = (tp + tn) / n
        boot['Sensitivity'].append(sens); boot['Specificity'].append(spec)
        boot['Accuracy'].append(acc);     boot['F1'].append(f1)
        boot['PPV'].append(prec);         boot['NPV'].append(npv)

    result = {}
    for k, vals in boot.items():
        vals = np.array(vals)
        if len(vals) == 0:
            result[k] = (np.nan, np.nan, np.nan)
        else:
            result[k] = (np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return result

def format_metrics_row(y_true, y_probs, threshold=0.5, n_bootstrap=1000, label=''):
    """返回含均值和95% CI的指标字典，用于保存到Excel。"""
    ci = bootstrap_metrics(y_true, y_probs, threshold=threshold, n_bootstrap=n_bootstrap)
    row = {'Dataset': label}
    for k, (mean, lo, hi) in ci.items():
        row[k]              = round(mean, 4)
        row[f'{k}_CI_low']  = round(lo, 4)
        row[f'{k}_CI_high'] = round(hi, 4)
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    row.update({'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
                'Threshold': threshold})
    return row

# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':

    BASE_DIR        = r'e:/RS/铜绿'
    TRAIN_DIR       = os.path.join(BASE_DIR, '皖北/皖北_混合_训练')
    INVAL_DIR       = os.path.join(BASE_DIR, '皖北/皖北_混合_验证')
    EXTVAL_DIR      = os.path.join(BASE_DIR, '分析_混合')
    RESULT_BASE_DIR = os.path.join(BASE_DIR, '建模结果')

    DRUGS = ['环丙沙星', '左氧氟沙星', '氨曲南', '庆大霉素', '妥布霉素', '头孢吡肟', '头孢他啶', '哌拉西林']

    DRUG_EN = {
        '环丙沙星':  'Ciprofloxacin',
        '左氧氟沙星': 'Levofloxacin',
        '氨曲南':    'Aztreonam',
        '庆大霉素':  'Gentamicin',
        '妥布霉素':  'Tobramycin',
        '头孢吡肟':  'Cefepime',
        '头孢他啶':  'Ceftazidime',
        '哌拉西林':  'Piperacillin',
    }

    ENABLE_HYPERPARAMETER_TUNING = True
    CV_FOLDS        = 5
    RANDOM_STATE    = 42
    ALPHA           = 0.05
    TOP_N_FEATURES       = 20   # SHAP 图显示 Top N
    ABLATION_MAX_FEATURES = 40  # 特征消融最多搜索 N 个特征
    SHAP_SAMPLE_SIZE = 200
    N_BOOTSTRAP     = 1000

    os.makedirs(RESULT_BASE_DIR, exist_ok=True)

    def get_models_config():
        return {
            'RF': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                'params': {'n_estimators': [100, 200], 'max_depth': [20, 50],
                           'min_samples_split': [2, 5], 'max_features': ['sqrt']},
            },
            'DT': {
                'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
                'params': {'max_depth': [20, 50], 'min_samples_split': [2, 10],
                           'min_samples_leaf': [1, 4], 'criterion': ['gini', 'entropy']},
            },
            'GB': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {'n_estimators': [100, 200], 'max_depth': [3, 5],
                           'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]},
            },
            'XGB': {
                'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1),
                'params': {'n_estimators': [100, 200], 'max_depth': [5, 7],
                           'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0],
                           'colsample_bytree': [0.8, 1.0]},
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=RANDOM_STATE),
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5, 1.0]},
            },
            'LR': {
                'model': LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, max_iter=2000),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']},
            },
            'NB': {
                'model': GaussianNB(),
                'params': {'var_smoothing': [1e-9, 1e-7, 1e-5]},
            },
            'LGBM': {
                'model': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
                'params': {'n_estimators': [100, 200], 'max_depth': [5, 7],
                           'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63],
                           'subsample': [0.8, 1.0]},
            }
        }

    all_drugs_summary = []  # 跨药物汇总

    for drug in DRUGS:
        drug_en = DRUG_EN[drug]
        print(f"\n{'='*80}\n药物: {drug} ({drug_en})\n{'='*80}")

        result_folder = os.path.join(RESULT_BASE_DIR, drug)
        os.makedirs(result_folder, exist_ok=True)

        # 加载数据
        train_df = pd.read_excel(os.path.join(TRAIN_DIR, f'{drug}.xlsx'))
        inval_df = pd.read_excel(os.path.join(INVAL_DIR, f'{drug}.xlsx'))
        extval_df = pd.read_excel(os.path.join(EXTVAL_DIR, f'{drug}.xlsx'))

        mz_cols = [c for c in train_df.columns if c.startswith('mz_')]
        X        = train_df[mz_cols];   y        = train_df['group'].astype(int)
        X_inval  = inval_df[mz_cols];   y_inval  = inval_df['group'].astype(int)
        X_extval = extval_df[mz_cols];  y_extval = extval_df['group'].astype(int)
        feature_names = mz_cols

        print(f"训练集: {X.shape[0]}行  {dict(y.value_counts())}")
        print(f"内部验证: {X_inval.shape[0]}行  {dict(y_inval.value_counts())}")
        print(f"外部验证: {X_extval.shape[0]}行  {dict(y_extval.value_counts())}")

        models_config = get_models_config()
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # ====================================================================
        # 1. 超参数调优 + 交叉验证（所有模型，含Bootstrap 95% CI）
        # ====================================================================
        best_params_dict  = {}
        all_model_results = {}
        trained_models    = {}

        for model_name, config in models_config.items():
            print(f"\n  [{drug}] {model_name} 训练中...")

            if ENABLE_HYPERPARAMETER_TUNING:
                gs = GridSearchCV(config['model'], config['params'],
                                  cv=skf, scoring='roc_auc', n_jobs=-1, verbose=0)
                gs.fit(X.values, y)
                best_params = gs.best_params_
                best_params_dict[model_name] = best_params
                print(f"    最佳参数: {best_params}  CV AUC={gs.best_score_:.4f}")
            else:
                best_params = {}
                best_params_dict[model_name] = '默认参数'

            # 全训练集最终模型
            model = deepcopy(config['model'])
            if best_params:
                model.set_params(**best_params)
            model.fit(X.values, y)
            trained_models[model_name] = {'model': model}

            # 交叉验证
            fold_rows = []
            all_yt, all_yp = [], []
            tprs_list = []
            base_fpr  = np.linspace(0, 1, 101)

            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
                X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
                y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
                mf = deepcopy(config['model'])
                if best_params:
                    mf.set_params(**best_params)
                mf.fit(X_tr, y_tr)
                yp_fold = mf.predict_proba(X_te)[:, 1]
                all_yt.extend(y_te); all_yp.extend(yp_fold)

                # 用Youden阈值（在全折合并后确定，此处暂用0.5，后续Bootstrap CI用opt_thr）
                ypred = (yp_fold >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_te, ypred, labels=[0,1]).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1   = 2*prec*sens/(prec+sens) if (prec+sens) > 0 else 0
                acc  = (tp+tn)/(tp+tn+fp+fn)
                fold_rows.append({'Fold': fold_idx,
                                  'AUC': roc_auc_score(y_te, yp_fold),
                                  'Accuracy': acc, 'Sensitivity': sens,
                                  'Specificity': spec, 'Precision': prec, 'F1': f1,
                                  'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)})
                fpr_c, tpr_c, _ = roc_curve(y_te, yp_fold)
                ti = np.interp(base_fpr, fpr_c, tpr_c); ti[0] = 0.0
                tprs_list.append(ti)

            # Bootstrap CI 基于全部折合并预测
            opt_thr, youden_val, _ = find_optimal_threshold(
                np.array(all_yt), np.array(all_yp), method='youden')
            ci_row = format_metrics_row(
                np.array(all_yt), np.array(all_yp),
                threshold=opt_thr, n_bootstrap=N_BOOTSTRAP, label='CV (Bootstrap)')

            fold_df = pd.DataFrame(fold_rows)
            mean_row = {'Fold': 'Mean±Std'}
            for k in ['AUC','Accuracy','Sensitivity','Specificity','Precision','F1']:
                mean_row[k] = fold_df[k].mean()
            for k in ['TP','TN','FP','FN']:
                mean_row[k] = fold_df[k].sum()
            fold_df = pd.concat([fold_df, pd.DataFrame([mean_row])], ignore_index=True)

            all_model_results[model_name] = {
                'results_df':        fold_df,
                'mean_auc':          mean_row['AUC'],
                'all_y_true':        np.array(all_yt),
                'all_y_probs':       np.array(all_yp),
                'tprs':              tprs_list,
                'base_fpr':          base_fpr,
                'optimal_threshold': opt_thr,
                'youden_index':      youden_val,
                'cv_ci_row':         ci_row,
            }
            print(f"    CV AUC={mean_row['AUC']:.4f}, Thr={opt_thr:.4f} | "
                  f"Bootstrap AUC={ci_row['AUC']:.4f} [{ci_row['AUC_CI_low']:.4f}-{ci_row['AUC_CI_high']:.4f}]")

        model_names = list(all_model_results.keys())

        # ====================================================================
        # 2. ROC 曲线图（所有模型，训练集CV）
        # ====================================================================
        colors_all = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
                      '#9467bd','#8c564b','#e377c2','#7f7f7f']

        plt.figure(figsize=(12, 10))
        for idx, mn in enumerate(model_names):
            r = all_model_results[mn]
            mean_tpr = np.mean(r['tprs'], axis=0); mean_tpr[-1] = 1.0
            mean_auc = auc_score(r['base_fpr'], mean_tpr)
            std_auc  = r['results_df'].iloc[:-1]['AUC'].std()
            std_tpr  = np.std(r['tprs'], axis=0)
            plt.plot(r['base_fpr'], mean_tpr, color=colors_all[idx], lw=2.5, alpha=0.8,
                     label=f'{mn} (AUC={mean_auc:.3f}±{std_auc:.3f})')
            plt.fill_between(r['base_fpr'],
                             np.maximum(mean_tpr - std_tpr, 0),
                             np.minimum(mean_tpr + std_tpr, 1),
                             color=colors_all[idx], alpha=0.1)
        plt.plot([0,1],[0,1],'--', lw=2, color='gray', alpha=0.5, label='Chance')
        plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
        plt.xlabel('FPR', fontsize=13); plt.ylabel('TPR', fontsize=13)
        plt.title(f'ROC Curves (5-Fold CV) - {drug_en}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(result_folder, 'roc_cv_all_models.pdf'),
                    format='pdf', bbox_inches='tight')
        plt.close()

        # 混淆矩阵图（所有模型，Youden阈值）
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for idx, mn in enumerate(model_names):
            r = all_model_results[mn]
            ypred = (r['all_y_probs'] >= r['optimal_threshold']).astype(int)
            cm = confusion_matrix(r['all_y_true'], ypred, labels=[0,1])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['S','NS'], yticklabels=['S','NS'], ax=axes[idx])
            acc = (cm[0,0]+cm[1,1])/cm.sum()
            axes[idx].set_title(f'{mn} (Acc={acc:.3f})', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted'); axes[idx].set_ylabel('True')
        plt.suptitle(f'Confusion Matrices (CV) - {drug_en}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, 'cm_cv_all_models.pdf'),
                    format='pdf', bbox_inches='tight')
        plt.close()

        # ====================================================================
        # 3. DeLong检验 → 筛选保留模型
        # ====================================================================
        best_model_name = max(all_model_results, key=lambda x: all_model_results[x]['mean_auc'])
        delong_rows     = []
        retained_models = [best_model_name]

        for other in model_names:
            if other == best_model_name:
                continue
            y_dl = all_model_results[best_model_name]['all_y_true']
            p, a1, a2 = delong_roc_test(y_dl,
                                         all_model_results[best_model_name]['all_y_probs'],
                                         all_model_results[other]['all_y_probs'])
            if p >= ALPHA:
                retained_models.append(other)
            delong_rows.append({
                'Model1': best_model_name, 'AUC1': round(a1,4),
                'Model2': other,           'AUC2': round(a2,4),
                'P_value': p,
                'Decision': f'保留(P>={ALPHA})' if p >= ALPHA else f'排除(P<{ALPHA})'
            })
        delong_df = pd.DataFrame(delong_rows)
        print(f"\n  [{drug}] 保留模型: {retained_models}")

        # ====================================================================
        # 4. SHAP（保留模型）
        # ====================================================================
        shap_feature_importance = {}

        for mn in retained_models:
            print(f"  [{drug}] SHAP: {mn}")
            model_shap = trained_models[mn]['model']
            X_arr = X.values
            n_s   = min(SHAP_SAMPLE_SIZE, X_arr.shape[0])
            np.random.seed(RANDOM_STATE)
            s_idx = np.random.choice(X_arr.shape[0], n_s, replace=False)
            X_s   = X_arr[s_idx]
            try:
                if mn in ['RF', 'XGB', 'DT', 'GB', 'LGBM']:
                    explainer = shap.TreeExplainer(model_shap)
                    sv_raw = explainer.shap_values(X_s)
                    # 二分类 TreeExplainer 可能返回:
                    #   list[cls0, cls1]          → 每元素 shape (n, f)
                    #   ndarray (2, n, f)         → axis0 = class
                    #   ndarray (n, f, 2)         → axis2 = class  (部分SHAP版本)
                    #   ndarray (n, f)            → XGB/LGBM 直接正类差值
                    if isinstance(sv_raw, list):
                        sv = np.array(sv_raw[1])          # 取正类
                    else:
                        sv = np.array(sv_raw)
                        if sv.ndim == 3:
                            if sv.shape[0] == n_s:        # (n, f, 2) 格式
                                sv = sv[:, :, 1]
                            else:                         # (2, n, f) 格式
                                sv = sv[1]
                        # ndim==2 时 XGB/LGBM 已是正类差值，直接使用
                else:
                    bg_n = min(100, n_s)
                    np.random.seed(RANDOM_STATE + 1)
                    bg_idx = np.random.choice(n_s, bg_n, replace=False)
                    bg = X_s[bg_idx]
                    _model = model_shap   # 固定闭包变量，避免 lambda 捕获最新迭代
                    explainer = shap.KernelExplainer(
                        lambda x: _model.predict_proba(x)[:, 1], bg)
                    sv = np.array(explainer.shap_values(X_s))
                    if sv.ndim == 3:
                        sv = sv[1]

                # 确保 sv 为 2D (n_samples, n_features)
                assert sv.ndim == 2 and sv.shape == (len(X_s), len(feature_names)), \
                    f"sv shape {sv.shape} 不符预期 ({len(X_s)}, {len(feature_names)})"

                feat_imp = np.abs(sv).mean(axis=0)
                imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': feat_imp}
                                      ).sort_values('Importance', ascending=False)
                shap_feature_importance[mn] = imp_df

                X_s_df = pd.DataFrame(X_s, columns=feature_names)
                y_s    = y.values[s_idx]

                # 统一构建 Explanation 对象（供 Waterfall / Scatter 复用）
                base_val = get_expected_value(explainer)
                explanation = shap.Explanation(
                    values=sv, base_values=base_val,
                    data=X_s, feature_names=feature_names)

                # 1. Beeswarm
                plt.figure(figsize=(12, 8))
                shap.summary_plot(sv, X_s_df, plot_type='dot',
                                  show=False, max_display=TOP_N_FEATURES)
                plt.title(f'SHAP Beeswarm - {mn} ({drug_en})', fontsize=13, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f'shap_beeswarm_{mn}.pdf'),
                            format='pdf', bbox_inches='tight')
                plt.close()

                # 2. Bar
                plt.figure(figsize=(10, 7))
                shap.summary_plot(sv, X_s_df, plot_type='bar',
                                  show=False, max_display=TOP_N_FEATURES)
                plt.title(f'SHAP Bar - {mn} ({drug_en})', fontsize=13, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f'shap_bar_{mn}.pdf'),
                            format='pdf', bbox_inches='tight')
                plt.close()

                # 3. Waterfall（各5个 NS/S 样本）
                try:
                    pos_idx = np.where(y_s == 1)[0]
                    neg_idx = np.where(y_s == 0)[0]
                    np.random.seed(RANDOM_STATE)
                    for label_tag, indices in [('NS', pos_idx), ('S', neg_idx)]:
                        chosen = np.random.choice(indices, min(5, len(indices)), replace=False)
                        for k, si in enumerate(chosen, 1):
                            plt.figure(figsize=(10, 8))
                            shap.plots.waterfall(explanation[si], show=False)
                            plt.title(f'SHAP Waterfall - {mn} {label_tag} #{k} ({drug_en})',
                                      fontsize=12, fontweight='bold')
                            plt.tight_layout()
                            plt.savefig(os.path.join(result_folder,
                                        f'shap_waterfall_{mn}_{label_tag}_{k}.pdf'),
                                        format='pdf', bbox_inches='tight')
                            plt.close()
                except Exception as e_wf:
                    print(f"    Waterfall失败: {e_wf}")

                # 4. Scatter（Top N 特征）
                try:
                    top_feats = imp_df.head(TOP_N_FEATURES)['Feature'].tolist()
                    feat_name_list = list(feature_names)
                    for feat in top_feats:
                        feat_idx = feat_name_list.index(feat)
                        plt.figure(figsize=(8, 6))
                        shap.plots.scatter(explanation[:, feat_idx], show=False)
                        plt.title(f'SHAP Scatter - {feat} - {mn} ({drug_en})',
                                  fontsize=11, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(os.path.join(result_folder,
                                    f'shap_scatter_{mn}_{feat}.pdf'),
                                    format='pdf', bbox_inches='tight')
                        plt.close()
                except Exception as e_sc:
                    print(f"    Scatter失败: {e_sc}")

                # 5. Force plot（HTML）
                try:
                    force_plot = shap.force_plot(base_val, sv, X_s_df,
                                                 feature_names=feature_names)
                    shap.save_html(
                        os.path.join(result_folder, f'shap_force_{mn}.html'),
                        force_plot)
                except Exception as e_fp:
                    print(f"    Force plot失败: {e_fp}")

                print(f"    SHAP完成 Top5: {', '.join(imp_df.head(5)['Feature'])}")
            except Exception as e:
                import traceback
                print(f"    SHAP失败: {e}")
                traceback.print_exc()

        # ====================================================================
        # 5. 特征消融（保留模型）
        # ====================================================================
        ablation_results = {}

        for mn in retained_models:
            if mn not in shap_feature_importance:
                continue
            print(f"  [{drug}] 特征消融: {mn}")
            # 消融最多取 ABLATION_MAX_FEATURES 个特征（可多于 SHAP 显示的 TOP_N_FEATURES）
            top_features = shap_feature_importance[mn].head(ABLATION_MAX_FEATURES)['Feature'].tolist()
            config = models_config[mn]
            auc_scores_ab, all_sub_probs = [], {}

            for n_feat in range(1, len(top_features) + 1):
                X_sub = X[top_features[:n_feat]]
                fold_aucs, sub_yt, sub_yp = [], [], []
                for tr_idx, te_idx in skf.split(X_sub, y):
                    mf = deepcopy(config['model'])
                    bp = best_params_dict.get(mn, {})
                    if isinstance(bp, dict) and bp:
                        mf.set_params(**bp)
                    mf.fit(X_sub.iloc[tr_idx].values, y.iloc[tr_idx])
                    yp = mf.predict_proba(X_sub.iloc[te_idx].values)[:, 1]
                    sub_yt.extend(y.iloc[te_idx]); sub_yp.extend(yp)
                    fold_aucs.append(roc_auc_score(y.iloc[te_idx], yp))
                auc_scores_ab.append(np.mean(fold_aucs))
                all_sub_probs[n_feat] = {'y_true': np.array(sub_yt), 'y_probs': np.array(sub_yp)}

            full_probs  = all_model_results[mn]['all_y_probs']
            full_ytrue  = all_model_results[mn]['all_y_true']
            full_auc    = all_model_results[mn]['mean_auc']
            ab_delong   = []
            opt_n       = None

            for n_feat in range(1, len(top_features) + 1):
                sd    = all_sub_probs[n_feat]
                s_auc = auc_scores_ab[n_feat - 1]
                try:
                    p, a1, a2 = delong_roc_test(full_ytrue, full_probs, sd['y_probs'])
                    sig = '有显著差异' if p < ALPHA else '无显著差异'
                except Exception:
                    auc_diff = abs(s_auc - full_auc)
                    sig = '有显著差异(估算)' if auc_diff > 0.05 else '无显著差异(估算)'
                    p   = 0.04 if auc_diff > 0.05 else 0.10
                ab_delong.append({'N_Features': n_feat, 'Subset_AUC': round(s_auc, 4),
                                  'Full_AUC': round(full_auc, 4), 'P_value': p, 'Significance': sig})
                if opt_n is None and p >= ALPHA:
                    opt_n = n_feat

            # 若所有特征数均与全特征有统计差异，取最大值（DeLong 最接近阈值处）
            if opt_n is None:
                opt_n = len(top_features)
                print(f"    警告: {mn} 在 {len(top_features)} 个特征内未找到非劣解，"
                      f"使用最大特征数 {opt_n}")

            ablation_results[mn] = {
                'feature_counts':    list(range(1, len(top_features) + 1)),
                'auc_scores':        auc_scores_ab,
                'top_features':      top_features,
                'delong_results':    pd.DataFrame(ab_delong),
                'optimal_n_features': opt_n,
                'optimal_features':  top_features[:opt_n],
                'opt_probs':         all_sub_probs[opt_n],   # 供跨模型 DeLong 使用
            }

            # 特征消融折线图
            plt.figure(figsize=(10, 7))
            plt.plot(list(range(1, len(top_features)+1)), auc_scores_ab,
                     marker='o', lw=2, markersize=6, color='steelblue', label=mn)
            opt_auc = auc_scores_ab[opt_n - 1]
            plt.scatter([opt_n], [opt_auc], s=200, marker='*', color='red',
                        edgecolors='black', lw=2, zorder=5, label=f'Optimal (n={opt_n})')
            plt.xlabel('Number of Features', fontsize=12)
            plt.ylabel('AUC', fontsize=12)
            plt.title(f'Feature Ablation - {mn} ({drug_en})', fontsize=13, fontweight='bold')
            plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(result_folder, f'ablation_{mn}.pdf'),
                        format='pdf', bbox_inches='tight')
            plt.close()
            print(f"    最优特征数: {opt_n}, AUC={opt_auc:.4f}")

        # ====================================================================
        # 6. 选最终模型
        #    Step A: 各保留模型的「最优特征版本」进行跨模型 DeLong 比较
        #    Step B: 保留与最优特征版本AUC最高者无统计差异的模型
        #    Step C: 从中选特征数最少的作为最终模型
        # ====================================================================
        final_model_name = None
        final_model_info = None
        candidates = {}
        for mn in retained_models:
            if mn in ablation_results:
                ab = ablation_results[mn]
                candidates[mn] = {
                    'n_features': ab['optimal_n_features'],
                    'features':   ab['optimal_features'],
                    'auc':        ab['auc_scores'][ab['optimal_n_features'] - 1],
                    'opt_probs':  ab['opt_probs'],
                }

        cross_delong_rows = []
        if candidates:
            # Step A: 找最优特征版本中AUC最高的参考模型
            ref_mn = max(candidates, key=lambda x: candidates[x]['auc'])
            ref_probs  = candidates[ref_mn]['opt_probs']['y_probs']
            ref_ytrue  = candidates[ref_mn]['opt_probs']['y_true']

            # Step B: 对每个候选模型的最优特征版本与参考模型做 DeLong 检验
            qualified = [ref_mn]  # 参考模型自身无需比较
            for mn in candidates:
                if mn == ref_mn:
                    cross_delong_rows.append({
                        'Model': mn, 'N_Features': candidates[mn]['n_features'],
                        'AUC': round(candidates[mn]['auc'], 4),
                        'Ref_Model': ref_mn, 'P_value': 1.0,
                        'Decision': '参考模型(保留)'
                    })
                    continue
                cand_probs = candidates[mn]['opt_probs']['y_probs']
                try:
                    p, a1, a2 = delong_roc_test(ref_ytrue, ref_probs, cand_probs)
                except Exception:
                    p = 0.0
                keep = p >= ALPHA
                if keep:
                    qualified.append(mn)
                cross_delong_rows.append({
                    'Model': mn, 'N_Features': candidates[mn]['n_features'],
                    'AUC': round(candidates[mn]['auc'], 4),
                    'Ref_Model': ref_mn, 'P_value': round(p, 4),
                    'Decision': f'保留(P>={ALPHA})' if keep else f'排除(P<{ALPHA})'
                })
                print(f"    跨模型DeLong [{mn} vs {ref_mn}]: P={p:.4f} → {'保留' if keep else '排除'}")

            # Step C: 在通过 DeLong 的候选中选特征数最少的
            qualified_candidates = {mn: candidates[mn] for mn in qualified}
            final_model_name = min(qualified_candidates, key=lambda x: qualified_candidates[x]['n_features'])
            final_model_info = qualified_candidates[final_model_name]
            print(f"\n  [{drug}] 最终模型: {final_model_name}  "
                  f"特征数={final_model_info['n_features']}  AUC={final_model_info['auc']:.4f}")

        # ====================================================================
        # 7. 内部验证 & 外部验证（含 Bootstrap 95% CI + 图表）
        # ====================================================================
        inval_row    = None
        extval_row   = None
        mic_corr_df  = pd.DataFrame()

        if final_model_name:
            sel  = final_model_info['features']
            config = models_config[final_model_name]
            final_model = deepcopy(config['model'])
            bp = best_params_dict.get(final_model_name, {})
            if isinstance(bp, dict) and bp:
                final_model.set_params(**bp)
            final_model.fit(X[sel].values, y)

            opt_thr = all_model_results[final_model_name]['optimal_threshold']

            y_iv_probs = final_model.predict_proba(X_inval[sel].values)[:, 1]
            y_ev_probs = final_model.predict_proba(X_extval[sel].values)[:, 1]

            inval_row  = format_metrics_row(y_inval,  y_iv_probs,
                                            threshold=opt_thr, n_bootstrap=N_BOOTSTRAP,
                                            label='内部验证')
            extval_row = format_metrics_row(y_extval, y_ev_probs,
                                            threshold=opt_thr, n_bootstrap=N_BOOTSTRAP,
                                            label='外部验证')
            inval_row['Drug']  = drug; inval_row['Model']  = final_model_name
            extval_row['Drug'] = drug; extval_row['Model'] = final_model_name

            print(f"  [{drug}] 内部验证  AUC={inval_row['AUC']:.4f} "
                  f"[{inval_row['AUC_CI_low']:.4f}-{inval_row['AUC_CI_high']:.4f}]  "
                  f"Sen={inval_row['Sensitivity']:.4f}  Spe={inval_row['Specificity']:.4f}")
            print(f"  [{drug}] 外部验证  AUC={extval_row['AUC']:.4f} "
                  f"[{extval_row['AUC_CI_low']:.4f}-{extval_row['AUC_CI_high']:.4f}]  "
                  f"Sen={extval_row['Sensitivity']:.4f}  Spe={extval_row['Specificity']:.4f}")

            for tag, y_t, y_proba in [
                ('internal_val', y_inval, y_iv_probs),
                ('external_val', y_extval, y_ev_probs)
            ]:
                label_cn = '内部验证' if 'internal' in tag else '外部验证'
                y_pred   = (y_proba >= opt_thr).astype(int)
                auc_val  = roc_auc_score(y_t, y_proba)
                ci_r     = bootstrap_metrics(y_t, y_proba, threshold=opt_thr,
                                             n_bootstrap=N_BOOTSTRAP)

                # ROC
                fpr_v, tpr_v, _ = roc_curve(y_t, y_proba)
                plt.figure(figsize=(8, 7))
                plt.plot(fpr_v, tpr_v, 'b', lw=2.5,
                         label=f'{final_model_name}\nAUC={auc_val:.3f} '
                               f'[{ci_r["AUC"][1]:.3f}-{ci_r["AUC"][2]:.3f}]')
                plt.plot([0,1],[0,1],'--', color='gray', lw=2, alpha=0.5)
                plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
                plt.xlabel('FPR', fontsize=13); plt.ylabel('TPR', fontsize=13)
                plt.title(f'ROC - {label_cn} - {drug_en}', fontsize=14, fontweight='bold')
                plt.legend(loc='lower right', fontsize=11)
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f'roc_{tag}.pdf'),
                            format='pdf', bbox_inches='tight')
                plt.close()

                # 混淆矩阵
                cm = confusion_matrix(y_t, y_pred, labels=[0,1])
                plt.figure(figsize=(7, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['S','NS'], yticklabels=['S','NS'])
                plt.xlabel('Predicted'); plt.ylabel('True')
                acc_v = (cm[0,0]+cm[1,1])/cm.sum()
                plt.title(f'Confusion Matrix - {label_cn} - {drug_en} (Acc={acc_v:.3f})',
                          fontsize=13, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f'cm_{tag}.pdf'),
                            format='pdf', bbox_inches='tight')
                plt.close()

                # PR曲线
                prec_c, rec_c, _ = precision_recall_curve(y_t, y_proba)
                pr_auc = auc_score(rec_c, prec_c)
                plt.figure(figsize=(8, 7))
                plt.plot(rec_c, prec_c, 'g', lw=2.5,
                         label=f'{final_model_name} (PR-AUC={pr_auc:.3f})')
                plt.xlabel('Recall', fontsize=13); plt.ylabel('Precision', fontsize=13)
                plt.title(f'PR Curve - {label_cn} - {drug_en}', fontsize=14, fontweight='bold')
                plt.legend(loc='upper right'); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f'pr_{tag}.pdf'),
                            format='pdf', bbox_inches='tight')
                plt.close()

                # DCA曲线
                thr_range = np.linspace(0.01, 0.99, 100)
                nbs = [calculate_net_benefit(y_t, y_proba, t) for t in thr_range]
                prev = np.mean(y_t)
                treat_all = [prev - (1 - prev) * (t / (1 - t)) for t in thr_range]
                plt.figure(figsize=(9, 7))
                plt.plot(thr_range, nbs, 'b', lw=2.5, label=final_model_name)
                plt.plot(thr_range, treat_all, 'k--', lw=2, alpha=0.5, label='Treat All')
                plt.plot(thr_range, [0]*100, color='gray', lw=2, alpha=0.5, label='Treat None')
                y_up = max(max(treat_all), max(nbs), 0.05) + 0.05
                plt.xlim([0,1]); plt.ylim([-0.05, y_up])
                plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit')
                plt.title(f'DCA - {label_cn} - {drug_en}', fontsize=14, fontweight='bold')
                plt.legend(loc='upper right'); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f'dca_{tag}.pdf'),
                            format='pdf', bbox_inches='tight')
                plt.close()

            # 保存模型pkl
            pkl_data = {
                'drug': drug, 'model_name': final_model_name, 'model': final_model,
                'features': sel, 'n_features': final_model_info['n_features'],
                'optimal_threshold': opt_thr
            }
            with open(os.path.join(result_folder, f'model_{final_model_name}.pkl'), 'wb') as f:
                pickle.dump(pkl_data, f)

        # ====================================================================
        # 7.5  特征峰强度 vs MIC 相关性分析（最终选定特征，仅 MIC 检测方法行）
        # ====================================================================
        if final_model_name and final_model_info:
            sel_feats = final_model_info['features']

            def _parse_mic(val):
                """将 MIC 值字符串转为 float，去掉 > < = ≤ ≥ 符号。"""
                if pd.isna(val):
                    return np.nan
                s = str(val).strip()
                for ch in ('>', '<', '=', '≤', '≥'):
                    s = s.replace(ch, '')
                try:
                    return float(s.strip())
                except ValueError:
                    return np.nan

            if '检测方法' in train_df.columns:
                mic_mask = train_df['检测方法'].str.upper().str.strip() == 'MIC'
            else:
                mic_mask = pd.Series([True] * len(train_df), index=train_df.index)
                print(f"  [{drug}] 训练集无'检测方法'列，使用全部行做MIC相关性分析")
            mic_train  = train_df[mic_mask].copy()
            mic_parsed = mic_train['耐药值'].apply(_parse_mic)
            valid_mask = mic_parsed.notna() & (mic_parsed > 0)
            mic_parsed = mic_parsed[valid_mask]
            mic_train  = mic_train[valid_mask]

            print(f"  [{drug}] MIC相关性分析: MIC有效样本数={len(mic_parsed)}, "
                  f"选定特征数={len(sel_feats)}")

            if len(mic_parsed) >= 10:
                mic_log = np.log2(mic_parsed.values.astype(float))  # log2 MIC
                X_mic   = mic_train[sel_feats].values
                group_s = mic_train['group'].values

                # ── Spearman 相关 ──────────────────────────────────────────
                corr_rows = []
                for fi, feat in enumerate(sel_feats):
                    r, p = stats.spearmanr(X_mic[:, fi], mic_log)
                    corr_rows.append({'Feature': feat, 'Spearman_r': r, 'P_value': p})

                corr_df  = pd.DataFrame(corr_rows)

                # ── BH-FDR 校正 ────────────────────────────────────────────
                pv       = corr_df['P_value'].values.copy()
                n_t      = len(pv)
                rank_ord = np.argsort(pv)
                adj      = pv[rank_ord] * n_t / (np.arange(1, n_t + 1))
                adj      = np.minimum(1.0, adj)
                for i in range(n_t - 2, -1, -1):
                    adj[i] = min(adj[i], adj[i + 1])
                p_adj = np.empty(n_t)
                p_adj[rank_ord] = adj

                corr_df['P_adj_BH']   = p_adj
                corr_df['Significant'] = corr_df['P_adj_BH'] < ALPHA
                corr_df['Spearman_r']  = corr_df['Spearman_r'].round(4)
                corr_df['P_value']     = corr_df['P_value'].round(6)
                corr_df['P_adj_BH']    = corr_df['P_adj_BH'].round(6)
                corr_df = corr_df.sort_values('P_adj_BH').reset_index(drop=True)
                mic_corr_df = corr_df

                sig_feats = corr_df[corr_df['Significant']]['Feature'].tolist()
                print(f"    显著相关（BH-FDR<0.05）: {len(sig_feats)}/{len(sel_feats)} 个特征")

                # ── 选定特征全览热图（所有选定特征 vs log₂MIC，按 r 排序）──
                acd = corr_df.sort_values('Spearman_r').reset_index(drop=True)
                n_sel = len(acd)
                # 自适应图宽：每个特征约 0.45 英寸，最小 8 最大 36
                fig_w = max(8, min(36, n_sel * 0.45))
                fig, ax = plt.subplots(figsize=(fig_w, 3.5))
                r_mat = acd['Spearman_r'].values.reshape(1, -1)
                sns.heatmap(r_mat, ax=ax,
                            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                            linewidths=0.5, annot=False,
                            cbar_kws={'label': 'Spearman r', 'shrink': 0.6},
                            xticklabels=acd['Feature'].tolist(),
                            yticklabels=[''])
                # 显著特征加 * 标注
                sig_pos = acd.index[acd['Significant']].tolist()
                for pos in sig_pos:
                    ax.text(pos + 0.5, 0.5, '*',
                            ha='center', va='center',
                            fontsize=8, color='black', fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
                n_sig_sel = int(acd['Significant'].sum())
                ax.set_title(
                    f'Selected {n_sel} Features vs log₂(MIC) - {drug_en}  '
                    f'(Spearman, sorted by r,  * = BH-FDR < 0.05,  '
                    f'n_sig={n_sig_sel},  n_MIC={len(mic_parsed)})',
                    fontsize=11, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(result_folder, 'mic_heatmap_all.pdf'),
                            format='pdf', bbox_inches='tight')
                plt.close()
                print(f"    选定特征热图已保存: mic_heatmap_all.pdf  "
                      f"(显著 {n_sig_sel}/{n_sel})")

                # ── 绘图：仅绘制显著特征 ───────────────────────────────────
                if sig_feats:
                    from matplotlib.patches import Patch
                    ncols  = min(4, len(sig_feats))
                    nrows  = (len(sig_feats) + ncols - 1) // ncols
                    fig, axes = plt.subplots(nrows, ncols,
                                             figsize=(5 * ncols, 4 * nrows),
                                             squeeze=False)
                    color_map = {0: '#2196F3', 1: '#F44336'}

                    for idx, feat in enumerate(sig_feats):
                        ax      = axes[idx // ncols][idx % ncols]
                        fi      = sel_feats.index(feat)
                        x_vals  = X_mic[:, fi]
                        info    = corr_df[corr_df['Feature'] == feat].iloc[0]

                        ax.scatter(x_vals, mic_log,
                                   c=[color_map[g] for g in group_s],
                                   alpha=0.6, s=30, edgecolors='none')

                        # 线性趋势线（log2 MIC ~ intensity）
                        z       = np.polyfit(x_vals, mic_log, 1)
                        x_line  = np.linspace(x_vals.min(), x_vals.max(), 100)
                        ax.plot(x_line, np.polyval(z, x_line),
                                'k--', lw=1.5, alpha=0.7)

                        ax.set_xlabel(feat, fontsize=9)
                        ax.set_ylabel('log₂(MIC)', fontsize=9)
                        ax.set_title(f'r={info["Spearman_r"]:.3f}  '
                                     f'P_adj={info["P_adj_BH"]:.4f}',
                                     fontsize=9)
                        ax.grid(True, alpha=0.3)

                    # 隐藏多余子图
                    for idx in range(len(sig_feats), nrows * ncols):
                        axes[idx // ncols][idx % ncols].set_visible(False)

                    legend_handles = [Patch(facecolor='#2196F3', label='S'),
                                      Patch(facecolor='#F44336', label='NS')]
                    fig.legend(handles=legend_handles, loc='lower right', fontsize=10)
                    plt.suptitle(
                        f'Peak Intensity vs log₂(MIC) - {drug_en}\n'
                        f'Spearman, BH-FDR < 0.05  (n={len(mic_parsed)})',
                        fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_folder, 'mic_correlation.pdf'),
                                format='pdf', bbox_inches='tight')
                    plt.close()
                    print(f"    MIC相关性图已保存: mic_correlation.pdf")
                else:
                    print(f"    无显著相关特征，跳过绘图")
            else:
                print(f"  [{drug}] MIC有效样本不足(n={len(mic_parsed)})，跳过相关性分析")

        # ====================================================================
        # 8. 保存所有结果到 results.xlsx
        # ====================================================================
        excel_path = os.path.join(result_folder, 'results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

            # 每模型每Fold原始指标
            for mn, r in all_model_results.items():
                r['results_df'].to_excel(writer, sheet_name=f'CV_{mn}', index=False)

            # CV Bootstrap CI 汇总（所有模型）
            cv_ci_rows = []
            for mn, r in all_model_results.items():
                row = {'Model': mn, **r['cv_ci_row'],
                       'Retained': 'Yes' if mn in retained_models else 'No',
                       'Is_Final': 'Yes' if mn == final_model_name else 'No'}
                cv_ci_rows.append(row)
            pd.DataFrame(cv_ci_rows).sort_values('AUC', ascending=False
                         ).to_excel(writer, sheet_name='CV_Bootstrap_CI', index=False)

            # DeLong（模型间，全特征）
            if len(delong_df) > 0:
                delong_df.to_excel(writer, sheet_name='DeLong_全特征', index=False)

            # DeLong（模型间，最优特征版本）
            if cross_delong_rows:
                pd.DataFrame(cross_delong_rows).to_excel(
                    writer, sheet_name='DeLong_最优特征', index=False)

            # 验证结果
            val_rows = []
            if inval_row:  val_rows.append(inval_row)
            if extval_row: val_rows.append(extval_row)
            if val_rows:
                pd.DataFrame(val_rows).to_excel(writer, sheet_name='验证结果_CI', index=False)

            # SHAP重要性
            for mn, imp_df in shap_feature_importance.items():
                imp_df.to_excel(writer, sheet_name=f'SHAP_{mn}', index=False)

            # 特征消融
            for mn, ab in ablation_results.items():
                pd.DataFrame({'N_Features': ab['feature_counts'],
                              'AUC': ab['auc_scores']}).to_excel(
                    writer, sheet_name=f'Ablation_{mn}', index=False)
                ab['delong_results'].to_excel(
                    writer, sheet_name=f'AblaDeLong_{mn}', index=False)

            # MIC 相关性（全部特征结果，含不显著）
            if not mic_corr_df.empty:
                mic_corr_df.to_excel(writer, sheet_name='MIC_相关性', index=False)

            # 最终模型汇总
            if final_model_name and final_model_info:
                pd.DataFrame([{
                    'Drug': drug, 'Final_Model': final_model_name,
                    'N_Features': final_model_info['n_features'],
                    'Features': ', '.join(final_model_info['features']),
                    'Train_AUC': final_model_info['auc']
                }]).to_excel(writer, sheet_name='最终模型', index=False)

        print(f"  [{drug}] 结果已保存: {excel_path}")

        # 收集跨药物汇总
        if inval_row and extval_row:
            for row in [inval_row, extval_row]:
                all_drugs_summary.append(row)

    # ========================================================================
    # 汇总所有药物验证结果
    # ========================================================================
    if all_drugs_summary:
        summary_df = pd.DataFrame(all_drugs_summary)
        out_path   = os.path.join(RESULT_BASE_DIR, '所有药物验证汇总.xlsx')
        summary_df.to_excel(out_path, index=False)
        print(f"\n汇总表已保存: {out_path}")
        cols_show = ['Drug','Dataset','Model','AUC','AUC_CI_low','AUC_CI_high',
                     'Sensitivity','Specificity','Accuracy','F1']
        print(summary_df[cols_show].to_string(index=False))

    print("\n✅ 所有药物建模完成！")
