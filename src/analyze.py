#!/usr/bin/env python3
"""Reproduce three METABRIC endocrine-therapy figures (Assignment 7)."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, chi2
from scipy.special import expit
from patsy import dmatrix

plt.switch_backend('Agg')

COLORS = {0: '#c77e5e', 1: '#397eaa'}
LABELS = {0: 'Likely premenopausal (≤45)', 1: 'Likely postmenopausal (≥55)'}
OUTCOME = 'hormone_therapy'


def proportion_table(data, stratifier=None):
    """Return treatment counts, proportions, and Wilson 95% confidence limits."""
    keys = ([stratifier] if stratifier else []) + ['older']
    result = (data.groupby(keys, observed=True)[OUTCOME]
              .agg(treated='sum', total='count').reset_index())
    result['rate'] = result.treated / result.total
    z = norm.ppf(0.975)
    n, p = result.total, result.rate
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    result['ci_low'] = center-half
    result['ci_high'] = center+half
    return result


def plot_grouped(ax, data, categories, col, title, small_n=10):
    """Grouped proportions with Wilson intervals and denominators."""
    summary = proportion_table(data, col)
    x = np.arange(len(categories))
    width = .37
    for age in (0, 1):
        sub = summary[summary.older == age].set_index(col)
        position = x + (age - .5)*width
        for i, (category, xp) in enumerate(zip(categories, position)):
            if category not in sub.index:
                continue
            row = sub.loc[category]
            rate = float(row.rate*100)
            lo = (row.rate-row.ci_low)*100
            hi = (row.ci_high-row.rate)*100
            ax.bar(xp, rate, width=width*.94, color=COLORS[age],
                   label=LABELS[age] if i == 0 else None)
            ax.errorbar(xp, rate, yerr=np.array([[lo], [hi]]),
                        color='#333333', capsize=3, fmt='none', lw=1.1)
            ax.text(xp, min(97, rate+hi+2.0), f'{rate:.0f}%\nn={int(row.total)}',
                    ha='center', va='bottom', fontsize=7.6,
                    color='#8c2b23' if row.total < small_n else '#222222')
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 111)
    ax.set_ylabel('Received endocrine therapy (%)')
    ax.set_title(title, fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    return summary


def cohort_gap_table(data):
    rows = []
    for cohort, sub in data.groupby('cohort', observed=True):
        young = sub[sub['older'] == 0]
        old = sub[sub['older'] == 1]
        p0 = young[OUTCOME].mean()
        p1 = old[OUTCOME].mean()
        n0 = len(young)
        n1 = len(old)
        se = np.sqrt(p0*(1-p0)/n0 + p1*(1-p1)/n1)
        gap = p1 - p0
        rows.append({
            'cohort': int(cohort),
            'n_young': n0,
            'n_older': n1,
            'young_rate': p0,
            'older_rate': p1,
            'gap': gap,
            'ci_low': gap - 1.96*se,
            'ci_high': gap + 1.96*se,
        })
    return pd.DataFrame(rows).sort_values('cohort').reset_index(drop=True)


def counterfactual_designs(fit, data):
    """Build model-aligned matrices without relying on statsmodels design_info."""
    d0 = data.copy()
    d1 = data.copy()
    d0['older'] = 0
    d1['older'] = 1
    rhs = fit.model.formula.split('~', 1)[1]
    columns = list(fit.params.index)
    X0 = dmatrix(rhs, d0, return_type='dataframe').reindex(columns=columns, fill_value=0)
    X1 = dmatrix(rhs, d1, return_type='dataframe').reindex(columns=columns, fill_value=0)
    return np.asarray(X0, dtype=float), np.asarray(X1, dtype=float)


def standardized_probabilities(fit, data):
    """Standardize predicted treatment rates to the observed covariate mix."""
    young = data.copy()
    older = data.copy()
    young['older'] = 0
    older['older'] = 1
    p_young = float(fit.predict(young).mean())
    p_older = float(fit.predict(older).mean())
    return {
        'prob_young': p_young,
        'prob_older': p_older,
        'gap': p_older - p_young,
    }


def make_model_dataset(df):
    model_df = df.copy()
    model_df['pr'] = (
        model_df['pr_status']
        .map({'Positive': 'PR+', 'Negative': 'PR−'})
        .fillna('Unknown')
    )
    model_df['pam50_all'] = (
        model_df['pam50_+_claudin-low_subtype']
        .fillna('Unknown')
        .replace({'NC': 'Unknown'})
    )
    model_df['tumor_stage_cat'] = (
        model_df['tumor_stage'].map({1: 'I', 2: 'II', 3: 'III'})
        .fillna('Unknown')
    )
    model_df['grade_cat'] = (
        model_df['neoplasm_histologic_grade'].map({1: '1', 2: '2', 3: '3'})
        .fillna('Unknown')
    )
    model_df['her2_cat'] = model_df['her2_status'].fillna('Unknown')
    model_df['chemo_cat'] = (
        pd.to_numeric(model_df['chemotherapy'], errors='coerce')
        .map({0: 'No', 1: 'Yes'})
        .fillna('Unknown')
    )
    model_df['cohort_cat'] = (
        model_df['cohort'].astype('Int64').astype(str)
        .replace({'<NA>': 'Unknown'})
    )
    return model_df[['older', OUTCOME, 'pr', 'pam50_all', 'tumor_stage_cat', 'grade_cat',
                     'her2_cat', 'chemo_cat', 'cohort_cat']].copy().reset_index(drop=True)


def get_formulas():
    return [
       ('Age group only', f'{OUTCOME} ~ older'),
       ('+ tumor biology', f'{OUTCOME} ~ older + C(pr) + C(pam50_all)'),
       ('+ clinical factors', f'{OUTCOME} ~ older + C(pr) + C(pam50_all)'
        ' + C(tumor_stage_cat) + C(grade_cat) + C(her2_cat) + C(chemo_cat)'),
       ('+ source cohort', f'{OUTCOME} ~ older + C(pr) + C(pam50_all)'
        ' + C(tumor_stage_cat) + C(grade_cat) + C(her2_cat) + C(chemo_cat) + C(cohort_cat)'),
       ('+ source cohort (no chemo)', f'{OUTCOME} ~ older + C(pr) + C(pam50_all)'
        ' + C(tumor_stage_cat) + C(grade_cat) + C(her2_cat) + C(cohort_cat)'),
    ]


def fit_sequential_models(mdf, out):
    formulas = get_formulas()
    estimates = []
    for j, (name, formula) in enumerate(formulas, 1):
        fit = smf.glm(formula, data=mdf, family=sm.families.Binomial()).fit()
        ci = fit.conf_int().loc['older']
        std = standardized_probabilities(fit, mdf)
        estimates.append({'model': name, 'n': len(mdf),
                          'odds_ratio': float(np.exp(fit.params['older'])),
                          'or_ci_low': float(np.exp(ci.iloc[0])),
                          'or_ci_high': float(np.exp(ci.iloc[1])),
                          'p_value': float(fit.pvalues['older']),
                          **std})
        summary_lines = fit.summary().as_text().splitlines()
        clean_summary = '\n'.join(line.rstrip() for line in summary_lines)
        (out / f'model_{j}_summary.txt').write_text(
            clean_summary + '\n'
        )
    results = pd.DataFrame(estimates)
    return results


def bootstrap_gap_changes(mdf, n_boot=40, seed=0):
    """Bootstrap uncertainty for standardized gaps and their reductions."""
    formulas = get_formulas()
    rng = np.random.default_rng(seed)
    arr = np.full((n_boot, len(formulas)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(mdf), len(mdf))
        boot = mdf.iloc[idx].copy()
        try:
            for j, (_, formula) in enumerate(formulas):
                fit = smf.glm(
                    formula, data=boot, family=sm.families.Binomial()
                ).fit(maxiter=35, disp=0)
                X0, X1 = counterfactual_designs(fit, boot)
                beta = np.asarray(fit.params, dtype=float)
                arr[b, j] = expit(X1 @ beta).mean() - expit(X0 @ beta).mean()
        except Exception:
            continue
    valid = arr[~np.isnan(arr).any(axis=1)]
    if len(valid) == 0:
        raise RuntimeError('No successful bootstrap samples')
    formulas_names = [name for name, _ in formulas]
    rows = []
    for j, name in enumerate(formulas_names):
        gap_vals = valid[:, j]
        reduce_vals = valid[:, 0] - valid[:, j]
        rows.append({
            'model': name,
            'bootstrap_n': len(valid),
            'gap_boot_ci_low': float(np.quantile(gap_vals, 0.025)),
            'gap_boot_ci_high': float(np.quantile(gap_vals, 0.975)),
            'reduction_boot_ci_low': float(np.quantile(reduce_vals, 0.025)),
            'reduction_boot_ci_high': float(np.quantile(reduce_vals, 0.975)),
        })
    return pd.DataFrame(rows)


def run(csv_path, output_dir, n_boot=40, seed=0):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path, low_memory=False)
    needed = ['er_status', 'age_at_diagnosis', OUTCOME, 'tumor_stage',
              'pr_status', 'pam50_+_claudin-low_subtype', 'cohort',
              'neoplasm_histologic_grade', 'her2_status', 'chemotherapy']
    missing = sorted(set(needed) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = data.loc[(data.er_status == 'Positive') &
                  ((data.age_at_diagnosis <= 45) |
                   (data.age_at_diagnosis >= 55)), needed].copy()
    df['older'] = (df.age_at_diagnosis >= 55).astype(int)
    df[OUTCOME] = pd.to_numeric(df[OUTCOME], errors='coerce')
    df = df.loc[df[OUTCOME].isin([0, 1])].copy()
    df['stage'] = df.tumor_stage.map({1: 'I', 2: 'II', 3: 'III'})

    # Figure 1
    overall = proportion_table(df)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5),
                             gridspec_kw={'width_ratios': [.8, 1.7]})
    ax = axes[0]
    for _, row in overall.iterrows():
        age = int(row.older)
        x = age
        ax.bar(x, 100*row.rate, color=COLORS[age], width=.63)
        ax.errorbar(x, 100*row.rate,
                    yerr=[[100*(row.rate-row.ci_low)], [100*(row.ci_high-row.rate)]],
                    fmt='none', capsize=5, color='#333333')
        ax.text(
            x, 100*row.ci_high+3,
            f'{100*row.rate:.1f}%\n({int(row.treated)}/{int(row.total)})',
            ha='center', fontsize=11)
    ax.set_xticks([0, 1], ['≤45', '≥55'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Received endocrine therapy (%)')
    ax.set_title('A  |  Overall treatment receipt', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    stage = plot_grouped(axes[1], df[df.stage.notna()], ['I', 'II', 'III'],
                         'stage', 'B  |  Treatment receipt within tumor stage')
    axes[1].set_xlabel('Tumor stage')
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[j]) for j in [0, 1]],
               [LABELS[j] for j in [0, 1]], loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .045), frameon=False)
    fig.suptitle(
        'Figure 1 | Endocrine therapy receipt among ER-positive patients',
        fontsize=16, y=.99
    )
    fig.text(
        .5, .008,
        'Whiskers: Wilson 95% CI  •  Age groups proxy menopause',
        ha='center', fontsize=9, color='#555555'
    )
    fig.tight_layout(rect=[0, .15, 1, .94])
    fig.savefig(out/'figure1_treatment_and_stage.png', dpi=210, bbox_inches='tight')
    plt.close(fig)
    overall.to_csv(out/'overall_treatment_rates.csv', index=False)
    stage.to_csv(out/'stage_treatment_rates.csv', index=False)

    # Figure 2
    cohort_df = df[df.cohort.isin([1, 2, 3, 4, 5])].copy()
    cohort_df['cohort'] = cohort_df['cohort'].astype(int)
    cohort_table = proportion_table(cohort_df, 'cohort')
    cohort_table.to_csv(out/'cohort_treatment_rates.csv', index=False)
    gaps = cohort_gap_table(cohort_df)
    gaps.to_csv(out/'cohort_absolute_gaps.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.7), gridspec_kw={'width_ratios': [1.55, 1.0]})
    plot_grouped(axes[0], cohort_df, [1, 2, 3, 4, 5], 'cohort', 'A  |  Treatment by source cohort')
    axes[0].set_xlabel('METABRIC cohort (identifier, not calendar year)')
    ay = axes[1]
    y = np.arange(len(gaps))[:: -1]
    vals = gaps['gap'].to_numpy()*100
    ay.errorbar(vals, y,
                xerr=[vals - gaps['ci_low'].to_numpy()*100,
                      gaps['ci_high'].to_numpy()*100 - vals],
                fmt='o', color=COLORS[1], capsize=3, markersize=6)
    ay.axvline(0, color='#666666', ls='--', lw=1)
    ay.set_yticks(y, [f'Cohort {c}' for c in gaps.cohort])
    ay.set_xlabel('Absolute treatment gap: ≥55 minus ≤45 (percentage points)')
    ay.set_title('B  |  Absolute treatment gap by cohort')
    ay.spines[['top', 'right']].set_visible(False)
    for yi, row in zip(y, gaps.itertuples()):
        ay.annotate(f'{row.gap*100:.0f} pp', (row.ci_high*100, yi), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=9)
    fig.suptitle(
        'Figure 2 | The treatment gap differs across METABRIC cohorts',
        fontsize=15, y=.98
    )
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[j]) for j in [0, 1]],
               [LABELS[j] for j in [0, 1]], loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .038), frameon=False)
    fig.text(
        .5, .012,
        'Wilson intervals for rates; approximate 95% intervals for gaps. '
        'Cohort ID is not a validated proxy for site or year.',
        fontsize=8.5, ha='center', color='#555555'
    )
    fig.tight_layout(rect=[0, .13, 1, .91], w_pad=3)
    fig.savefig(out/'figure2_cohort_heterogeneity.png', dpi=210, bbox_inches='tight')
    plt.close(fig)

    # Figure 3 and sensitivity analysis
    mdf = make_model_dataset(df)
    results = fit_sequential_models(mdf, out)
    boot = bootstrap_gap_changes(mdf, n_boot=n_boot, seed=seed)
    results = results.merge(boot, on='model', how='left')
    results['gap_reduction'] = results.loc[0, 'gap'] - results['gap']
    results.to_csv(out/'sequential_model_results.csv', index=False)

    # Cohort interaction test
    base = smf.glm(
        f'{OUTCOME} ~ older + C(cohort)', data=cohort_df,
        family=sm.families.Binomial()
    ).fit()
    interaction = smf.glm(
        f'{OUTCOME} ~ older * C(cohort)', data=cohort_df,
        family=sm.families.Binomial()
    ).fit()
    lr = 2*(interaction.llf-base.llf)
    df_diff = int(interaction.df_model-base.df_model)
    interaction_p = float(chi2.sf(lr, df_diff))
    (out/'cohort_interaction_test.txt').write_text(
        f'Likelihood-ratio test: age group x cohort\n'
        f'eligible_n={len(cohort_df)}\nLR={lr:.6f}\n'
        f'df={df_diff}\np={interaction_p:.8g}\n')

    # Plot Figure 3 with panel B showing how little the gap shrinks.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2),
                             gridspec_kw={'width_ratios': [1.35, 1.0]})
    ax = axes[0]
    main = results.iloc[:4].copy()
    yy = np.arange(len(main))[:: -1]
    gaps_pp = main['gap'].to_numpy()*100
    low_pp = main['gap_boot_ci_low'].to_numpy()*100
    high_pp = main['gap_boot_ci_high'].to_numpy()*100
    ax.errorbar(gaps_pp, yy, xerr=[gaps_pp-low_pp, high_pp-gaps_pp],
                fmt='o', color=COLORS[1], markersize=9, capsize=4, lw=2)
    ax.set_yticks(yy, main['model'])
    ax.set_xlabel('Standardized treatment gap (≥55 − ≤45, percentage points)')
    ax.set_title('A  |  Adjusted treatment gap')
    ax.set_xlim(16, 40.5)
    ax.grid(axis='x', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    for yi, row in zip(yy, main.itertuples()):
        ax.text(1.02, yi,
                f'{row.prob_young*100:.1f}% vs {row.prob_older*100:.1f}%  '
                f'(gap {row.gap*100:.1f} pp)',
                transform=ax.get_yaxis_transform(), ha='left',
                va='center', fontsize=9.8, clip_on=False)

    bx = axes[1]
    reduction_df = results.iloc[1:].copy()  # omit baseline, whose reduction is zero by definition
    reduction_df = reduction_df.iloc[[0, 1, 2, 3]]
    display_labels = [
        'Tumor biology', 'Clinical + chemo',
        'Source cohort + chemo', 'Source cohort, no chemo'
    ]
    reduction_pp = reduction_df['gap_reduction'].to_numpy()*100
    red_low = reduction_df['reduction_boot_ci_low'].to_numpy()*100
    red_high = reduction_df['reduction_boot_ci_high'].to_numpy()*100
    yy2 = np.arange(len(reduction_df))[:: -1]
    bx.errorbar(reduction_pp, yy2, xerr=[reduction_pp-red_low, red_high-reduction_pp],
                fmt='o', color='#7a4f9a', markersize=8, capsize=4, lw=2)
    bx.axvline(0, color='#666666', ls='--', lw=1)
    bx.set_yticks(yy2, display_labels)
    bx.set_xlabel('Reduction from unadjusted gap (pp)')
    bx.set_title('B  |  How much smaller is the gap?')
    bx.set_xlim(-4.5, 11.5)
    bx.grid(axis='x', alpha=.18)
    bx.spines[['top', 'right']].set_visible(False)
    for yi, row in zip(yy2, reduction_df.itertuples()):
        bx.annotate(f'{row.gap_reduction*100:.1f} pp', (row.reduction_boot_ci_high*100, yi),
                    xytext=(6, 0), textcoords='offset points', va='center', fontsize=9)

    fig.suptitle(
        f'Figure 3 | Does adjustment explain the treatment gap? (n={len(mdf)})',
        fontsize=16, y=.98
    )
    fig.text(.5, .035,
             'Whiskers: paired patient-bootstrap 95% percentile intervals. '
             'Age groups proxy menopause; associations are not causal.',
             ha='center', fontsize=8.7, color='#555555')
    fig.tight_layout(rect=[0, .09, 1, .93], w_pad=4)
    fig.savefig(out/'figure3_sequential_adjustment.png', dpi=210, bbox_inches='tight')
    plt.close(fig)

    print(f'Eligible ER+ cohort n={len(df)}; regression sample n={len(mdf)}')
    cols = [
        'model', 'gap', 'gap_boot_ci_low', 'gap_boot_ci_high',
        'gap_reduction', 'reduction_boot_ci_low',
        'reduction_boot_ci_high'
    ]
    print('\nSequential models:\n', results[cols].to_string(index=False))
    print(f'\nFigures and tables: {out.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='results/figures')
    parser.add_argument('--bootstrap', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error('--bootstrap must be at least 20 for useful intervals')
    run(args.input, args.output, n_boot=args.bootstrap, seed=args.seed)
