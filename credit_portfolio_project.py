"""
=============================================================
  INTEGRATED CREDIT PORTFOLIO MANAGEMENT SYSTEM
  Bank Internship Project — All 5 Modules Combined
=============================================================

WHAT THIS PROJECT DOES:
------------------------
This Python script simulates a real bank's credit risk system.
It covers all 5 key credit risk tasks:

  Module 1 → Credit Scorecard (who might default?)
  Module 2 → ECL under IFRS 9 (how much loss to expect?)
  Module 3 → Monte Carlo Simulation (worst-case exposure?)
  Module 4 → Stress Testing (what if economy crashes?)
  Module 5 → Migration Matrix (how do ratings change over time?)

HOW TO READ THIS FILE:
-----------------------
Every section has plain English comments explaining WHAT is
happening and WHY. Read those comments — they are as important
as the code itself for your internship presentation.
"""

# ─────────────────────────────────────────────────────────────
# STEP 0: IMPORT LIBRARIES
# Libraries are pre-built toolkits. Think of them like
# importing Excel's formula engine before using formulas.
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)  # Makes results reproducible — same output every run


# ─────────────────────────────────────────────────────────────
# STEP 1: GENERATE REALISTIC LOAN PORTFOLIO DATA
#
# In a real bank internship, you'd get this from the core
# banking system (Finacle, Temenos, etc.). Here we create
# 2,000 synthetic borrowers that mimic real loan books.
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  CREDIT PORTFOLIO MANAGEMENT SYSTEM")
print("  Generating loan portfolio data...")
print("=" * 60)

n = 2000  # Number of borrowers in our portfolio

# --- Borrower characteristics ---
age             = np.random.randint(22, 65, n)
income          = np.random.normal(55000, 20000, n).clip(15000, 200000)
loan_amount     = np.random.normal(40000, 15000, n).clip(5000, 150000)
loan_tenure_yr  = np.random.choice([1, 2, 3, 5, 7, 10], n)
interest_rate   = np.random.uniform(7.5, 18.0, n)          # % per annum
credit_score    = np.random.normal(680, 80, n).clip(300, 850)
debt_to_income  = loan_amount / income                      # DTI ratio
employment_yrs  = np.random.exponential(5, n).clip(0, 35)
num_past_dues   = np.random.poisson(0.4, n).clip(0, 5)

# --- Assign credit ratings (like S&P / internal bank ratings) ---
# Banks classify borrowers from AAA (best) to CCC (worst)
def assign_rating(score):
    if score >= 780: return 'AAA'
    elif score >= 740: return 'AA'
    elif score >= 700: return 'A'
    elif score >= 660: return 'BBB'
    elif score >= 620: return 'BB'
    elif score >= 580: return 'B'
    else: return 'CCC'

ratings = np.array([assign_rating(s) for s in credit_score])

# --- Simulate actual defaults (1 = defaulted, 0 = paid) ---
# Better credit score, lower income ratio → lower default probability
# This formula is NOT random — it mimics real-world credit behaviour
log_odds = (-0.008 * credit_score
            + 2.5 * debt_to_income
            + 0.4 * num_past_dues
            - 0.02 * employment_yrs
            + 1.2)
true_pd   = 1 / (1 + np.exp(-log_odds))
default   = (np.random.rand(n) < true_pd).astype(int)

# --- Assemble into a DataFrame (like an Excel table in Python) ---
df = pd.DataFrame({
    'borrower_id'    : [f'BRW{str(i).zfill(5)}' for i in range(n)],
    'age'            : age,
    'income'         : income.round(0),
    'loan_amount'    : loan_amount.round(0),
    'loan_tenure_yr' : loan_tenure_yr,
    'interest_rate'  : interest_rate.round(2),
    'credit_score'   : credit_score.round(0),
    'debt_to_income' : debt_to_income.round(4),
    'employment_yrs' : employment_yrs.round(1),
    'num_past_dues'  : num_past_dues,
    'rating'         : ratings,
    'default'        : default
})

print(f"\n  Portfolio size    : {n:,} borrowers")
print(f"  Total exposure    : ₹{df['loan_amount'].sum()/1e7:.1f} Cr")
print(f"  Observed defaults : {default.sum()} ({default.mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# MODULE 1: CREDIT SCORECARD — PD MODEL
#
# WHAT IS THIS?
# A scorecard is a statistical model that scores each borrower
# from 300-850. Low score = high risk. Banks use this to
# decide: approve loan? At what interest rate?
#
# HOW IT WORKS:
# We use Logistic Regression — a standard ML algorithm that
# outputs a probability between 0 and 1 (the Probability of
# Default, or PD).
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODULE 1: CREDIT SCORECARD (PD Model)")
print("─" * 60)

features = ['credit_score', 'debt_to_income', 'employment_yrs',
            'num_past_dues', 'income', 'loan_amount', 'interest_rate']
X = df[features]
y = df['default']

# Split: 70% train the model, 30% test it (standard practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardise: put all variables on the same scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

# Get PD scores for ALL borrowers (we need these in later modules)
df['PD'] = model.predict_proba(scaler.transform(X))[:, 1]

# Model performance metrics
auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
print(f"\n  Model AUC Score   : {auc:.3f}  (above 0.70 = good, 0.80 = very good)")
print(f"  Gini Coefficient  : {(2*auc-1):.3f}  (industry standard: >0.40 is acceptable)")

# What does AUC mean?
# AUC = 1.0 → perfect model. AUC = 0.5 → random guessing.
# Banks typically accept AUC > 0.70 for retail scoring.

# Score bands — translate PD to scorecard points (like CIBIL score logic)
df['score_band'] = pd.cut(df['PD'],
    bins=[0, 0.05, 0.10, 0.20, 0.35, 1.0],
    labels=['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])


# ─────────────────────────────────────────────────────────────
# MODULE 2: EXPECTED CREDIT LOSS (ECL) UNDER IFRS 9
#
# WHAT IS IFRS 9?
# It's a global accounting standard that requires banks to
# PROVISION (set aside money) for expected loan losses.
# Before IFRS 9 (pre-2018), banks only provisioned AFTER
# a loan went bad. IFRS 9 forces forward-looking provisioning.
#
# THE THREE STAGES:
# Stage 1 → Performing loans (no significant increase in risk)
#           Provision = 12-month ECL
# Stage 2 → Watch list (significant increase in credit risk)
#           Provision = Lifetime ECL
# Stage 3 → Non-performing / Defaulted loans
#           Provision = Lifetime ECL (higher LGD)
#
# ECL FORMULA:
# ECL = PD × LGD × EAD
#   PD  = Probability of Default (from Module 1)
#   LGD = Loss Given Default (how much we lose if they default)
#   EAD = Exposure at Default (outstanding loan balance)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODULE 2: ECL CALCULATION UNDER IFRS 9")
print("─" * 60)

# --- IFRS 9 Stage Classification ---
# Stage based on PD thresholds (simplified — real banks use
# 'significant increase in credit risk' criteria)
def classify_stage(pd_val, past_dues):
    if pd_val >= 0.20 or past_dues >= 3:
        return 3   # Defaulted / Credit-impaired
    elif pd_val >= 0.08 or past_dues >= 1:
        return 2   # Significant increase in credit risk
    else:
        return 1   # Performing

df['IFRS9_stage'] = df.apply(lambda r: classify_stage(r['PD'], r['num_past_dues']), axis=1)

# --- LGD: Loss Given Default ---
# Banks estimate LGD based on collateral, seniority, recovery history.
# Here: secured loans (lower LGD) vs unsecured (higher LGD)
# Stage 3 has higher LGD because recovery is harder at that point.
lgd_map = {1: 0.35, 2: 0.45, 3: 0.65}
df['LGD'] = df['IFRS9_stage'].map(lgd_map)
# Add some loan-level variation
df['LGD'] = (df['LGD'] + np.random.normal(0, 0.03, n)).clip(0.10, 0.90)

# --- EAD: Exposure at Default ---
# For term loans: outstanding principal.
# We simulate partial repayment based on tenure elapsed.
tenure_elapsed = np.random.uniform(0.1, 1.0, n)
df['EAD'] = df['loan_amount'] * (1 - 0.3 * tenure_elapsed)

# --- Lifetime PD for Stage 2 and 3 ---
# 12-month PD is what the model gives us.
# Lifetime PD ≈ 1 - (1 - PD_annual)^tenure  [simplified]
df['PD_lifetime'] = 1 - (1 - df['PD']) ** df['loan_tenure_yr']

# --- ECL Calculation ---
# Stage 1: 12-month ECL
# Stage 2 & 3: Lifetime ECL
df['ECL'] = np.where(
    df['IFRS9_stage'] == 1,
    df['PD']          * df['LGD'] * df['EAD'],   # 12-month
    df['PD_lifetime'] * df['LGD'] * df['EAD']    # Lifetime
)

# --- Macro Overlay: Apply 3 scenarios (Base, Adverse, Severe) ---
# IFRS 9 requires probability-weighted multiple economic scenarios
scenarios = {
    'Base'    : {'pd_mult': 1.0,  'weight': 0.50},
    'Adverse' : {'pd_mult': 1.4,  'weight': 0.35},
    'Severe'  : {'pd_mult': 2.2,  'weight': 0.15},
}
weighted_ecl = sum(
    df['ECL'] * s['pd_mult'] * s['weight']
    for s in scenarios.values()
)
df['ECL_scenario_weighted'] = weighted_ecl

total_ecl   = df['ECL_scenario_weighted'].sum()
total_ead   = df['EAD'].sum()
coverage    = total_ecl / total_ead * 100

stage_summary = df.groupby('IFRS9_stage').agg(
    count=('borrower_id', 'count'),
    total_ead=('EAD', 'sum'),
    total_ecl=('ECL_scenario_weighted', 'sum')
).reset_index()
stage_summary['ecl_rate'] = stage_summary['total_ecl'] / stage_summary['total_ead'] * 100

print(f"\n  IFRS 9 Stage Distribution:")
for _, row in stage_summary.iterrows():
    stage_names = {1: 'Stage 1 (Performing)', 2: 'Stage 2 (Watch)', 3: 'Stage 3 (Default)'}
    print(f"    {stage_names[int(row['IFRS9_stage'])]:<26}: {int(row['count']):>4} loans  "
          f"ECL Rate: {row['ecl_rate']:.1f}%")
print(f"\n  Total Portfolio EAD  : ₹{total_ead/1e7:.2f} Cr")
print(f"  Total Weighted ECL   : ₹{total_ecl/1e7:.2f} Cr")
print(f"  Coverage Ratio       : {coverage:.2f}%")


# ─────────────────────────────────────────────────────────────
# MODULE 3: MONTE CARLO SIMULATION — CREDIT EXPOSURE
#
# WHAT IS THIS?
# For large corporate borrowers (not retail), banks need to
# know: "In the WORST CASE, how much could we lose?"
# 
# Monte Carlo = run 10,000 random future scenarios and see
# the distribution of outcomes. Like weather forecasting —
# you can't predict exactly, but you can say "95% chance of
# less than X mm rain".
#
# KEY OUTPUT: PFE (Potential Future Exposure) at 95th percentile.
# Banks use this to set credit limits for counterparties.
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODULE 3: MONTE CARLO — POTENTIAL FUTURE EXPOSURE")
print("─" * 60)

n_sims     = 10000  # number of simulation paths
n_periods  = 12     # monthly steps (1 year horizon)
n_corp     = 50     # top 50 corporate borrowers in portfolio

# Pick top 50 loans by size as "corporate" counterparties
corp_df = df.nlargest(n_corp, 'loan_amount').copy()

# Simulate exposure paths for each corporate borrower
# Exposure follows a mean-reverting random walk (Vasicek-style):
# E(t+1) = E(t) * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)
mu      = -0.02   # mean repayment trend (exposure falls over time)
sigma   = 0.15    # volatility of exposure
dt      = 1/12    # monthly steps

pfe_results = []
for _, corp in corp_df.iterrows():
    E0       = corp['EAD']
    paths    = np.zeros((n_sims, n_periods))
    paths[:, 0] = E0
    for t in range(1, n_periods):
        Z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    paths = paths.clip(0)  # exposure cannot be negative
    pfe_95 = np.percentile(paths, 95, axis=0)  # 95th percentile path
    pfe_results.append({
        'borrower_id'   : corp['borrower_id'],
        'initial_ead'   : E0,
        'peak_pfe_95'   : pfe_95.max(),
        'pfe_month6'    : pfe_95[5],
        'pfe_month12'   : pfe_95[11],
        'expected_exp'  : paths.mean(axis=0).max(),
    })

pfe_df = pd.DataFrame(pfe_results)
print(f"\n  Corporate counterparties simulated : {n_corp}")
print(f"  Simulations per counterparty       : {n_sims:,}")
print(f"  Average Peak PFE (95th pctile)     : ₹{pfe_df['peak_pfe_95'].mean():,.0f}")
print(f"  Max Peak PFE in portfolio          : ₹{pfe_df['peak_pfe_95'].max():,.0f}")
print(f"  Borrowers exceeding current EAD    : "
      f"{(pfe_df['peak_pfe_95'] > pfe_df['initial_ead']).sum()} "
      f"(need credit limit review)")


# ─────────────────────────────────────────────────────────────
# MODULE 4: CREDIT PORTFOLIO STRESS TESTING
#
# WHAT IS THIS?
# Regulators (RBI in India, ECB in Europe) require banks to
# prove they can survive adverse economic conditions.
# Stress testing answers: "If GDP falls by 3%, how much MORE
# do we need to provision?"
#
# THE SCENARIOS WE TEST:
# Base Case    → Normal economy (no change)
# Mild Stress  → GDP -2%, unemployment +2% (2020-lite)
# Severe Shock → GDP -5%, unemployment +5% (2008-style crisis)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODULE 4: PORTFOLIO STRESS TESTING")
print("─" * 60)

stress_scenarios = {
    'Base Case': {
        'pd_multiplier'  : 1.00,
        'lgd_multiplier' : 1.00,
        'gdp_change'     : 0.0,
        'unemp_change'   : 0.0,
    },
    'Mild Stress': {
        'pd_multiplier'  : 1.35,   # PDs rise 35% in mild stress
        'lgd_multiplier' : 1.10,   # LGD rises 10% (collateral values fall)
        'gdp_change'     : -2.0,
        'unemp_change'   : +2.0,
    },
    'Severe Shock': {
        'pd_multiplier'  : 2.10,   # PDs more than double in severe stress
        'lgd_multiplier' : 1.25,   # LGD rises 25%
        'gdp_change'     : -5.0,
        'unemp_change'   : +5.0,
    },
}

stress_results = {}
for scenario_name, params in stress_scenarios.items():
    stressed_pd  = (df['PD']  * params['pd_multiplier']).clip(0, 0.99)
    stressed_lgd = (df['LGD'] * params['lgd_multiplier']).clip(0, 0.95)
    stressed_ecl = stressed_pd * stressed_lgd * df['EAD']

    stress_results[scenario_name] = {
        'total_ecl'     : stressed_ecl.sum(),
        'avg_pd'        : stressed_pd.mean(),
        'avg_lgd'       : stressed_lgd.mean(),
        'default_count' : (stressed_pd > 0.5).sum(),
        'ecl_change'    : stressed_ecl.sum() - df['ECL_scenario_weighted'].sum(),
        'gdp_change'    : params['gdp_change'],
        'unemp_change'  : params['unemp_change'],
    }

print(f"\n  Scenario Results:")
print(f"  {'Scenario':<16} {'Total ECL':>12} {'vs Base':>12} {'Avg PD':>8} {'Defaults':>10}")
print(f"  {'─'*16} {'─'*12} {'─'*12} {'─'*8} {'─'*10}")
base_ecl = stress_results['Base Case']['total_ecl']
for name, res in stress_results.items():
    delta = res['total_ecl'] - base_ecl
    delta_str = f"+₹{delta/1e6:.1f}M" if delta >= 0 else f"-₹{abs(delta)/1e6:.1f}M"
    print(f"  {name:<16} ₹{res['total_ecl']/1e6:>9.1f}M {delta_str:>12} "
          f"{res['avg_pd']:>7.1%} {res['default_count']:>10,}")


# ─────────────────────────────────────────────────────────────
# MODULE 5: CREDIT MIGRATION MATRIX
#
# WHAT IS THIS?
# Rating migration = tracking how borrowers move between
# credit rating categories over time.
# Example: A borrower rated 'A' today might be 'BBB' next year.
#
# WHY IT MATTERS:
# - Predicts future ECL under stress
# - Helps set risk appetite limits
# - Required for ICAAP (Internal Capital Adequacy Assessment)
#
# THE MATRIX: Rows = current rating, Columns = next year rating.
# Each cell = % of borrowers who transitioned that way.
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODULE 5: CREDIT MIGRATION MATRIX")
print("─" * 60)

rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'DEFAULT']

# Simulate next-year ratings based on PD and current rating
def simulate_next_rating(current_rating, pd_val):
    """
    Simulate where a borrower migrates to in 1 year.
    Higher PD = higher probability of downgrade.
    """
    rating_levels = {'AAA':7,'AA':6,'A':5,'BBB':4,'BB':3,'B':2,'CCC':1}
    level = rating_levels.get(current_rating, 4)

    # Default check first
    if np.random.rand() < pd_val:
        return 'DEFAULT'

    # Downgrade probability increases with PD
    downgrade_prob = pd_val * 3.5
    upgrade_prob   = (1 - pd_val) * 0.08

    rand = np.random.rand()
    if rand < downgrade_prob and level > 1:
        steps = np.random.choice([1, 2], p=[0.75, 0.25])
        new_level = max(1, level - steps)
    elif rand < downgrade_prob + upgrade_prob and level < 7:
        new_level = level + 1
    else:
        new_level = level

    level_to_rating = {v: k for k, v in rating_levels.items()}
    return level_to_rating[new_level]

df['next_rating'] = df.apply(
    lambda r: simulate_next_rating(r['rating'], r['PD']), axis=1
)

# Build the migration matrix
migration_matrix = pd.crosstab(
    df['rating'], df['next_rating'],
    normalize='index'  # rows sum to 1 (100%)
).round(4) * 100  # convert to percentages

# Reorder to standard rating order
migration_matrix = migration_matrix.reindex(
    index=[r for r in rating_order if r in migration_matrix.index],
    columns=[r for r in rating_order if r in migration_matrix.columns],
    fill_value=0
)

print("\n  Migration Matrix (% of borrowers moving from row → column rating):")
print(f"\n  {'':6}", end='')
for col in migration_matrix.columns:
    print(f"{col:>8}", end='')
print()
print(f"  {'─'*6}", '─'*8*len(migration_matrix.columns))
for idx, row in migration_matrix.iterrows():
    print(f"  {idx:<6}", end='')
    for col in migration_matrix.columns:
        val = row.get(col, 0)
        if col == idx:  # diagonal (stayed same rating)
            print(f"\033[1m{val:>7.1f}%\033[0m", end='')
        else:
            print(f"{val:>7.1f}%", end='')
    print()

# Key insight: diagonal values = % that STAYED in same rating
# Off-diagonal = % that migrated (up = good, down = bad)
stable_pct = np.mean([migration_matrix.loc[r, r]
                      for r in migration_matrix.index
                      if r in migration_matrix.columns
                      and r != 'DEFAULT'])
print(f"\n  Average stability (stayed in same rating): {stable_pct:.1f}%")


# ─────────────────────────────────────────────────────────────
# FINAL OUTPUT: GENERATE ALL CHARTS IN ONE DASHBOARD
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Generating Risk Dashboard Charts...")
print("─" * 60)

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.38)

BLUE   = '#1A56DB'
RED    = '#E02424'
GREEN  = '#057A55'
AMBER  = '#C27803'
PURPLE = '#7E3AF2'
DARK   = '#1F2937'
LIGHT  = '#F3F4F6'

def style_ax(ax, title):
    ax.set_facecolor('#FFFFFF')
    ax.set_title(title, fontsize=12, fontweight='bold', color=DARK, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.tick_params(colors='#4B5563', labelsize=9)
    ax.yaxis.label.set_color('#4B5563')
    ax.xaxis.label.set_color('#4B5563')

# ── Chart 1: PD Distribution (Module 1) ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['PD'], bins=40, color=BLUE, edgecolor='white', linewidth=0.5, alpha=0.85)
ax1.axvline(df['PD'].mean(), color=RED, linestyle='--', linewidth=1.5, label=f"Mean PD: {df['PD'].mean():.1%}")
ax1.set_xlabel('Probability of Default')
ax1.set_ylabel('Number of Borrowers')
ax1.legend(fontsize=9)
style_ax(ax1, 'M1 — PD Distribution')

# ── Chart 2: Score Band Breakdown (Module 1) ──────────────────
ax2 = fig.add_subplot(gs[0, 1])
band_counts = df['score_band'].value_counts()
colors_band = [GREEN, '#34D399', AMBER, '#F97316', RED]
bars = ax2.bar(range(len(band_counts)), band_counts.values,
               color=colors_band[:len(band_counts)], edgecolor='white')
ax2.set_xticks(range(len(band_counts)))
ax2.set_xticklabels([b.replace(' Risk', '') for b in band_counts.index], rotation=25, ha='right', fontsize=8)
ax2.set_ylabel('Borrowers')
for bar, val in zip(bars, band_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', va='bottom', fontsize=8, color=DARK)
style_ax(ax2, 'M1 — Risk Score Bands')

# ── Chart 3: IFRS 9 Stage ECL (Module 2) ─────────────────────
ax3 = fig.add_subplot(gs[0, 2])
stage_labels = ['Stage 1\nPerforming', 'Stage 2\nWatch List', 'Stage 3\nDefaulted']
stage_ecls   = [stage_summary.loc[stage_summary['IFRS9_stage']==s, 'total_ecl'].values[0] / 1e6
                for s in [1, 2, 3] if s in stage_summary['IFRS9_stage'].values]
stage_cols   = [GREEN, AMBER, RED]
bars3 = ax3.bar(stage_labels[:len(stage_ecls)], stage_ecls, color=stage_cols[:len(stage_ecls)], edgecolor='white')
ax3.set_ylabel('ECL (₹ Millions)')
for bar, val in zip(bars3, stage_ecls):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'₹{val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK)
style_ax(ax3, 'M2 — ECL by IFRS 9 Stage')

# ── Chart 4: Scenario-Weighted ECL Breakdown (Module 2) ───────
ax4 = fig.add_subplot(gs[1, 0])
scen_labels = list(scenarios.keys())
scen_ecls   = [df['ECL'] * s['pd_mult'] * s['weight'] for s in scenarios.values()]
scen_totals = [e.sum() / 1e6 for e in scen_ecls]
bars4 = ax4.bar(scen_labels, scen_totals,
                color=[GREEN, AMBER, RED], edgecolor='white')
ax4.set_ylabel('ECL Contribution (₹M)')
for bar, val in zip(bars4, scen_totals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'₹{val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK)
style_ax(ax4, 'M2 — Scenario-Weighted ECL')

# ── Chart 5: Monte Carlo PFE Paths (Module 3) ────────────────
ax5 = fig.add_subplot(gs[1, 1])
# Rerun one borrower's paths for display
sample_corp  = df.nlargest(1, 'loan_amount').iloc[0]
E0_disp      = sample_corp['EAD']
paths_disp   = np.zeros((200, n_periods))
paths_disp[:, 0] = E0_disp
for t in range(1, n_periods):
    Z = np.random.standard_normal(200)
    paths_disp[:, t] = paths_disp[:, t-1] * np.exp(
        (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
paths_disp = paths_disp.clip(0)
months = range(1, n_periods+1)
for path in paths_disp[:50]:
    ax5.plot(months, path/1e3, color=BLUE, alpha=0.08, linewidth=0.8)
ax5.plot(months, np.percentile(paths_disp, 95, axis=0)/1e3,
         color=RED, linewidth=2, label='PFE 95th pctile')
ax5.plot(months, paths_disp.mean(axis=0)/1e3,
         color=GREEN, linewidth=2, label='Expected exposure')
ax5.set_xlabel('Month')
ax5.set_ylabel("Exposure ('000 ₹)")
ax5.legend(fontsize=8)
style_ax(ax5, 'M3 — Monte Carlo Exposure Paths')

# ── Chart 6: PFE Distribution across Corporates (Module 3) ───
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(pfe_df['peak_pfe_95']/1e3, bins=20, color=PURPLE,
         edgecolor='white', linewidth=0.5, alpha=0.85)
ax6.axvline(pfe_df['peak_pfe_95'].mean()/1e3, color=RED, linestyle='--',
            linewidth=1.5, label=f"Mean: ₹{pfe_df['peak_pfe_95'].mean()/1e3:.0f}K")
ax6.set_xlabel("Peak PFE ('000 ₹)")
ax6.set_ylabel('Number of Counterparties')
ax6.legend(fontsize=8)
style_ax(ax6, 'M3 — PFE Distribution (50 Corporates)')

# ── Chart 7: Stress Testing ECL Comparison (Module 4) ────────
ax7 = fig.add_subplot(gs[2, 0])
stress_names = list(stress_results.keys())
stress_ecls  = [stress_results[s]['total_ecl']/1e6 for s in stress_names]
bars7 = ax7.bar(stress_names, stress_ecls,
                color=[GREEN, AMBER, RED], edgecolor='white', width=0.5)
ax7.set_ylabel('Total ECL (₹ Millions)')
for bar, val in zip(bars7, stress_ecls):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'₹{val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK)
style_ax(ax7, 'M4 — Stress Testing: ECL by Scenario')

# ── Chart 8: Stress — Default Count (Module 4) ───────────────
ax8 = fig.add_subplot(gs[2, 1])
stress_defaults = [stress_results[s]['default_count'] for s in stress_names]
bars8 = ax8.bar(stress_names, stress_defaults,
                color=[GREEN, AMBER, RED], edgecolor='white', width=0.5)
ax8.set_ylabel('Projected Defaults')
for bar, val in zip(bars8, stress_defaults):
    ax8.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             str(val), ha='center', va='bottom', fontsize=10, fontweight='bold', color=DARK)
style_ax(ax8, 'M4 — Stress Testing: Projected Defaults')

# ── Chart 9: ECL Waterfall — Base to Severe (Module 4) ───────
ax9 = fig.add_subplot(gs[2, 2])
base_ecl_val    = stress_results['Base Case']['total_ecl'] / 1e6
mild_inc        = (stress_results['Mild Stress']['total_ecl'] -
                   stress_results['Base Case']['total_ecl']) / 1e6
severe_inc      = (stress_results['Severe Shock']['total_ecl'] -
                   stress_results['Mild Stress']['total_ecl']) / 1e6
severe_total    = stress_results['Severe Shock']['total_ecl'] / 1e6
cats  = ['Base\nECL', 'Mild\nStress Δ', 'Severe\nΔ', 'Total\nSevere']
vals  = [base_ecl_val, mild_inc, severe_inc, severe_total]
cols  = [GREEN, AMBER, RED, '#7F1D1D']
ax9.bar(cats, vals, color=cols, edgecolor='white')
for i, (c, v) in enumerate(zip(cats, vals)):
    ax9.text(i, v + 0.05, f'₹{v:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold', color=DARK)
ax9.set_ylabel('₹ Millions')
style_ax(ax9, 'M4 — ECL Waterfall')

# ── Chart 10: Migration Matrix Heatmap (Module 5) ─────────────
ax10 = fig.add_subplot(gs[3, :])
mig_data = migration_matrix.values
cmap = LinearSegmentedColormap.from_list('risk', ['#F0FDF4', '#DCFCE7', '#FEF9C3', '#FEE2E2', '#7F1D1D'])
im = ax10.imshow(mig_data, cmap=cmap, aspect='auto', vmin=0, vmax=70)
ax10.set_xticks(range(len(migration_matrix.columns)))
ax10.set_xticklabels(migration_matrix.columns, fontsize=9)
ax10.set_yticks(range(len(migration_matrix.index)))
ax10.set_yticklabels(migration_matrix.index, fontsize=9)
ax10.set_xlabel('Rating Next Year →', fontsize=10)
ax10.set_ylabel('Rating This Year ↓', fontsize=10)
for i in range(len(migration_matrix.index)):
    for j in range(len(migration_matrix.columns)):
        val = mig_data[i, j]
        ax10.text(j, i, f'{val:.0f}%', ha='center', va='center',
                  fontsize=8, color='#1F2937' if val < 50 else 'white', fontweight='bold')
plt.colorbar(im, ax=ax10, shrink=0.6, label='Transition Probability (%)')
style_ax(ax10, 'M5 — Credit Migration Matrix (% Transition Probabilities)')

# ── Main Title ────────────────────────────────────────────────
fig.suptitle('Integrated Credit Portfolio Risk Management Dashboard\nBank Internship Project — All 5 Modules',
             fontsize=16, fontweight='bold', color=DARK, y=0.98)

plt.savefig('/mnt/user-data/outputs/credit_risk_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("\n  Dashboard saved!")


# ─────────────────────────────────────────────────────────────
# EXPORT EVERYTHING TO EXCEL (Management Report)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Exporting Excel Report...")
print("─" * 60)

with pd.ExcelWriter('/mnt/user-data/outputs/credit_portfolio_report.xlsx', engine='openpyxl') as writer:

    # Sheet 1: Full portfolio
    out_cols = ['borrower_id', 'age', 'income', 'loan_amount', 'credit_score',
                'rating', 'PD', 'score_band', 'IFRS9_stage', 'LGD', 'EAD',
                'ECL', 'ECL_scenario_weighted', 'next_rating']
    df[out_cols].to_excel(writer, sheet_name='Portfolio Data', index=False)

    # Sheet 2: IFRS 9 stage summary
    stage_summary.to_excel(writer, sheet_name='IFRS9 ECL Summary', index=False)

    # Sheet 3: Stress testing
    stress_df = pd.DataFrame(stress_results).T.reset_index()
    stress_df.columns = ['Scenario', 'Total ECL', 'Avg PD', 'Avg LGD',
                          'Default Count', 'ECL Change vs Base', 'GDP Change %', 'Unemp Change %']
    stress_df.to_excel(writer, sheet_name='Stress Testing', index=False)

    # Sheet 4: Monte Carlo PFE
    pfe_df.to_excel(writer, sheet_name='Monte Carlo PFE', index=False)

    # Sheet 5: Migration matrix
    migration_matrix.to_excel(writer, sheet_name='Migration Matrix')

print("  Excel report saved!")

print("\n" + "=" * 60)
print("  ALL MODULES COMPLETE")
print("  Output files:")
print("  1. credit_risk_dashboard.png  — 10-chart visual dashboard")
print("  2. credit_portfolio_report.xlsx — Full data + analysis")
print("=" * 60)
