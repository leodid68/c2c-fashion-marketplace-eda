"""
C2C Fashion Marketplace — Exploratory Data Analysis
=====================================================
Dataset: C2C Fashion Store User Data (Kaggle)
Author: Léo Didier
Date: March 2026

Goal: Explore 99k users from a C2C fashion marketplace to understand
engagement patterns, geographic dynamics, and find the story for a
Tableau dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (14, 6)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {name}.png")


# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FIRST LOOK
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. DATA LOADING & FIRST LOOK")
print("=" * 60)

df = pd.read_csv("data/6M-0K-99K.users.dataset.public.csv")
print(f"\nShape: {df.shape[0]:,} users × {df.shape[1]} features")
print(f"\nColumns:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nNull values:\n{df.isnull().sum().to_string()}")
print(f"\nDuplicates: {df.duplicated().sum()}")

# Drop useless columns
df = df.drop(columns=["index", "identifierHash", "type"])
print(f"\nAfter cleanup: {df.shape[1]} features")


# ══════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY & OUTLIERS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. DATA QUALITY & OUTLIERS")
print("=" * 60)

numeric_cols = [
    "socialNbFollowers", "socialNbFollows", "socialProductsLiked",
    "productsListed", "productsSold", "productsPassRate",
    "productsWished", "productsBought", "daysSinceLastLogin",
    "seniority", "seniorityAsMonths", "seniorityAsYears"
]

print("\n── Descriptive Statistics ──")
print(df[numeric_cols].describe().round(2).to_string())

# Outlier detection with IQR
print("\n── Outlier Detection (IQR method) ──")
outlier_report = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct = n_outliers / len(df) * 100
    outlier_report.append({
        "feature": col,
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower": lower, "upper": upper,
        "n_outliers": n_outliers,
        "pct_outliers": round(pct, 2)
    })
    if n_outliers > 0:
        print(f"  {col}: {n_outliers:,} outliers ({pct:.1f}%) — range [{lower:.0f}, {upper:.0f}]")

# Visualize distributions
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    ax = axes[i]
    data = df[col]
    # Use log scale for highly skewed distributions
    if data.max() > 100 and data.median() < 10:
        data_plot = data[data > 0]
        ax.hist(data_plot, bins=50, color="#2D31FA", alpha=0.7, edgecolor="white")
        ax.set_xscale("log")
        ax.set_title(f"{col}\n(log scale, zeros excluded)", fontsize=9)
    else:
        ax.hist(data, bins=50, color="#2D31FA", alpha=0.7, edgecolor="white")
        ax.set_title(col, fontsize=9)
    ax.tick_params(labelsize=7)
save_fig("01_distributions")


# ══════════════════════════════════════════════════════════════════════
# 3. ENGAGEMENT ANALYSIS — THE BIG STORY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ENGAGEMENT ANALYSIS")
print("=" * 60)

# Activity flags
df["is_seller"] = df["productsSold"] > 0
df["is_buyer"] = df["productsBought"] > 0
df["is_lister"] = df["productsListed"] > 0
df["is_social"] = df["socialProductsLiked"] > 0
df["is_active"] = df["is_seller"] | df["is_buyer"]

total = len(df)
print(f"\n── Activation Funnel ──")
print(f"  Total users:      {total:>7,} (100%)")
print(f"  Has profile pic:  {df.hasProfilePicture.sum():>7,} ({df.hasProfilePicture.mean()*100:.1f}%)")
print(f"  Social active:    {df.is_social.sum():>7,} ({df.is_social.mean()*100:.1f}%)")
print(f"  Listed ≥1:        {df.is_lister.sum():>7,} ({df.is_lister.mean()*100:.1f}%)")
print(f"  Bought ≥1:        {df.is_buyer.sum():>7,} ({df.is_buyer.mean()*100:.1f}%)")
print(f"  Sold ≥1:          {df.is_seller.sum():>7,} ({df.is_seller.mean()*100:.1f}%)")
print(f"  Any transaction:  {df.is_active.sum():>7,} ({df.is_active.mean()*100:.1f}%)")

# Funnel visualization
funnel_data = {
    "Registered": total,
    "Profile Picture": df.hasProfilePicture.sum(),
    "Liked ≥1 product": df.is_social.sum(),
    "Listed ≥1 product": df.is_lister.sum(),
    "Bought ≥1": df.is_buyer.sum(),
    "Sold ≥1": df.is_seller.sum(),
}
fig, ax = plt.subplots(figsize=(12, 6))
labels = list(funnel_data.keys())
values = list(funnel_data.values())
pcts = [v / total * 100 for v in values]
colors = ["#2D31FA", "#4444FF", "#6666FF", "#FF4D00", "#FF7733", "#FF9966"]
bars = ax.barh(labels[::-1], pcts[::-1], color=colors[::-1], edgecolor="white", height=0.6)
for bar, pct, val in zip(bars, pcts[::-1], values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}% ({val:,})", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("% of total users")
ax.set_title("C2C Fashion Marketplace — User Activation Funnel", fontsize=14, fontweight="bold")
ax.set_xlim(0, 115)
save_fig("02_activation_funnel")


# ══════════════════════════════════════════════════════════════════════
# 4. USER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. USER SEGMENTATION")
print("=" * 60)

def segment_user(row):
    sold = row["productsSold"]
    bought = row["productsBought"]
    liked = row["socialProductsLiked"]
    if sold >= 10 or bought >= 10:
        return "Power User"
    elif sold > 0 and bought > 0:
        return "Two-Way Trader"
    elif sold > 0:
        return "Seller Only"
    elif bought > 0:
        return "Buyer Only"
    elif liked > 0 or row["productsListed"] > 0:
        return "Browser"
    else:
        return "Ghost"

df["segment"] = df.apply(segment_user, axis=1)

print("\n── User Segments ──")
seg_counts = df["segment"].value_counts()
for seg, count in seg_counts.items():
    print(f"  {seg:<18} {count:>7,} ({count/total*100:>5.1f}%)")

# Segment chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
seg_order = ["Ghost", "Browser", "Buyer Only", "Seller Only", "Two-Way Trader", "Power User"]
seg_colors = ["#E0E0E0", "#B0B0FF", "#4444FF", "#FF9966", "#FF4D00", "#2D31FA"]
seg_vals = [seg_counts.get(s, 0) for s in seg_order]
explode = [0, 0, 0.05, 0.05, 0.1, 0.15]
ax1.pie(seg_vals, labels=seg_order, autopct="%1.1f%%", colors=seg_colors,
        explode=explode, startangle=90, textprops={"fontsize": 9})
ax1.set_title("User Segments Distribution", fontsize=13, fontweight="bold")

# Segment profile — mean metrics
seg_profile = df.groupby("segment")[
    ["socialNbFollowers", "socialProductsLiked", "productsListed", "productsSold", "productsBought"]
].mean().reindex(seg_order)
seg_profile.plot(kind="bar", ax=ax2, width=0.7)
ax2.set_title("Average Metrics by Segment", fontsize=13, fontweight="bold")
ax2.set_ylabel("Mean count")
ax2.tick_params(axis="x", rotation=30)
ax2.legend(fontsize=8, loc="upper left")
save_fig("03_user_segments")


# ══════════════════════════════════════════════════════════════════════
# 5. GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. GEOGRAPHIC ANALYSIS")
print("=" * 60)

# Top 15 countries
top_countries = df["country"].value_counts().head(15)
print("\n── Top 15 Countries ──")
for country, count in top_countries.items():
    print(f"  {country:<20} {count:>6,} ({count/total*100:.1f}%)")

# Activation rate by country (top 15)
country_stats = df[df["country"].isin(top_countries.index)].groupby("country").agg(
    total_users=("is_active", "count"),
    activation_rate=("is_active", "mean"),
    avg_sold=("productsSold", "mean"),
    avg_bought=("productsBought", "mean"),
    avg_followers=("socialNbFollowers", "mean"),
    ios_rate=("hasIosApp", "mean"),
    pct_female=("gender", lambda x: (x == "F").mean()),
).sort_values("total_users", ascending=False)

print("\n── Activation Rate by Country (top 15) ──")
for country, row in country_stats.iterrows():
    print(f"  {country:<20} users={row.total_users:>5,.0f}  activation={row.activation_rate*100:>5.1f}%  "
          f"iOS={row.ios_rate*100:.0f}%  female={row.pct_female*100:.0f}%")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Users by country
ax = axes[0]
top15 = country_stats.sort_values("total_users")
ax.barh(top15.index, top15["total_users"], color="#2D31FA", alpha=0.8)
ax.set_title("Users by Country", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of users")

# Activation rate by country
ax = axes[1]
top15_act = country_stats.sort_values("activation_rate")
colors = ["#FF4D00" if v > country_stats["activation_rate"].median() else "#B0B0FF"
          for v in top15_act["activation_rate"]]
ax.barh(top15_act.index, top15_act["activation_rate"] * 100, color=colors)
ax.set_title("Activation Rate by Country", fontsize=12, fontweight="bold")
ax.set_xlabel("% users with ≥1 transaction")
ax.axvline(country_stats["activation_rate"].median() * 100, color="gray", linestyle="--", alpha=0.5)

# iOS adoption by country
ax = axes[2]
top15_ios = country_stats.sort_values("ios_rate")
ax.barh(top15_ios.index, top15_ios["ios_rate"] * 100, color="#FF9966", alpha=0.8)
ax.set_title("iOS App Adoption by Country", fontsize=12, fontweight="bold")
ax.set_xlabel("% users with iOS app")

save_fig("04_geographic_analysis")


# ══════════════════════════════════════════════════════════════════════
# 6. GENDER DYNAMICS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. GENDER DYNAMICS")
print("=" * 60)

gender_stats = df.groupby("gender").agg(
    count=("gender", "size"),
    activation_rate=("is_active", "mean"),
    avg_sold=("productsSold", "mean"),
    avg_bought=("productsBought", "mean"),
    avg_listed=("productsListed", "mean"),
    avg_liked=("socialProductsLiked", "mean"),
    avg_followers=("socialNbFollowers", "mean"),
    ios_rate=("hasIosApp", "mean"),
    android_rate=("hasAndroidApp", "mean"),
).round(4)

print(f"\n{gender_stats.to_string()}")

# Gender by segment
gender_seg = pd.crosstab(df["segment"], df["gender"], normalize="columns") * 100
print("\n── Segment Distribution by Gender (%) ──")
print(gender_seg.round(1).to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gender split
gender_seg.reindex(seg_order).plot(kind="barh", ax=ax1, color=["#FF4D00", "#2D31FA"], width=0.6)
ax1.set_title("Segment Distribution by Gender", fontsize=12, fontweight="bold")
ax1.set_xlabel("% of gender group")
ax1.legend(title="Gender")

# App usage by gender
app_data = gender_stats[["ios_rate", "android_rate"]] * 100
app_data.plot(kind="bar", ax=ax2, color=["#2D31FA", "#FF4D00"], width=0.5)
ax2.set_title("App Adoption by Gender", fontsize=12, fontweight="bold")
ax2.set_ylabel("% users")
ax2.tick_params(axis="x", rotation=0)
ax2.legend(["iOS", "Android"])
save_fig("05_gender_dynamics")


# ══════════════════════════════════════════════════════════════════════
# 7. APP ADOPTION & ITS IMPACT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. APP ADOPTION & IMPACT")
print("=" * 60)

app_impact = df.groupby("hasAnyApp").agg(
    count=("hasAnyApp", "size"),
    activation_rate=("is_active", "mean"),
    avg_sold=("productsSold", "mean"),
    avg_bought=("productsBought", "mean"),
    avg_liked=("socialProductsLiked", "mean"),
    avg_followers=("socialNbFollowers", "mean"),
).round(4)
app_impact.index = ["No App", "Has App"]
print(f"\n{app_impact.to_string()}")

# Platform comparison
platform_data = []
for label, col in [("iOS", "hasIosApp"), ("Android", "hasAndroidApp")]:
    group = df[df[col] == True]
    platform_data.append({
        "platform": label,
        "users": len(group),
        "activation_rate": group["is_active"].mean(),
        "avg_sold": group["productsSold"].mean(),
        "avg_bought": group["productsBought"].mean(),
    })
platform_df = pd.DataFrame(platform_data)
print(f"\n── Platform Comparison ──")
print(platform_df.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# App vs No App
metrics = ["activation_rate", "avg_sold", "avg_bought", "avg_liked"]
labels = ["Activation Rate", "Avg Sold", "Avg Bought", "Avg Liked"]
x = np.arange(len(labels))
w = 0.3
no_app = [app_impact.loc["No App", m] for m in metrics]
has_app = [app_impact.loc["Has App", m] for m in metrics]
# Normalize for comparison
max_vals = [max(a, b) for a, b in zip(no_app, has_app)]
no_app_n = [a / m if m > 0 else 0 for a, m in zip(no_app, max_vals)]
has_app_n = [a / m if m > 0 else 0 for a, m in zip(has_app, max_vals)]
ax1.bar(x - w/2, no_app_n, w, label="No App", color="#B0B0FF")
ax1.bar(x + w/2, has_app_n, w, label="Has App", color="#2D31FA")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_title("App Users vs Non-App Users (normalized)", fontsize=12, fontweight="bold")
ax1.legend()
# Add actual values
for i, (na, ha) in enumerate(zip(no_app, has_app)):
    ax1.text(i - w/2, no_app_n[i] + 0.02, f"{na:.3f}", ha="center", fontsize=7)
    ax1.text(i + w/2, has_app_n[i] + 0.02, f"{ha:.3f}", ha="center", fontsize=7)

# iOS vs Android
ax2.bar(["iOS Users", "Android Users"],
        [platform_df.iloc[0]["activation_rate"] * 100, platform_df.iloc[1]["activation_rate"] * 100],
        color=["#2D31FA", "#FF4D00"], width=0.4)
ax2.set_title("Activation Rate: iOS vs Android", fontsize=12, fontweight="bold")
ax2.set_ylabel("% users with ≥1 transaction")
save_fig("06_app_impact")


# ══════════════════════════════════════════════════════════════════════
# 8. POWER USERS DEEP DIVE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. POWER USERS DEEP DIVE")
print("=" * 60)

power = df[df["segment"] == "Power User"]
print(f"\nPower users: {len(power):,} ({len(power)/total*100:.2f}%)")
print(f"\n── Power User Profile ──")
print(f"  Avg products sold:    {power.productsSold.mean():.1f}")
print(f"  Avg products bought:  {power.productsBought.mean():.1f}")
print(f"  Avg products listed:  {power.productsListed.mean():.1f}")
print(f"  Avg followers:        {power.socialNbFollowers.mean():.1f}")
print(f"  Avg liked:            {power.socialProductsLiked.mean():.1f}")
print(f"  Has app:              {power.hasAnyApp.mean()*100:.1f}%")
print(f"  iOS:                  {power.hasIosApp.mean()*100:.1f}%")
print(f"  Female:               {(power.gender=='F').mean()*100:.1f}%")

# Top countries for power users
power_countries = power["country"].value_counts().head(10)
print(f"\n── Top Countries for Power Users ──")
for country, count in power_countries.items():
    total_in_country = len(df[df["country"] == country])
    print(f"  {country:<20} {count:>4} power users ({count/total_in_country*100:.2f}% of country)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Power user activity scatter
active = df[df["is_active"]]
ax1.scatter(active["productsSold"], active["productsBought"],
            alpha=0.3, s=10, c="#2D31FA", edgecolors="none")
ax1.scatter(power["productsSold"], power["productsBought"],
            alpha=0.6, s=20, c="#FF4D00", edgecolors="none", label="Power Users")
ax1.set_xlabel("Products Sold")
ax1.set_ylabel("Products Bought")
ax1.set_title("Active Users: Sold vs Bought", fontsize=12, fontweight="bold")
ax1.legend()

# Power users by country (% of country)
power_rate = (
    df.groupby("country")["segment"]
    .apply(lambda x: (x == "Power User").mean() * 100)
    .sort_values(ascending=False)
    .head(15)
)
power_rate.sort_values().plot(kind="barh", ax=ax2, color="#FF4D00", alpha=0.8)
ax2.set_title("Power User Rate by Country (top 15)", fontsize=12, fontweight="bold")
ax2.set_xlabel("% of users who are Power Users")
save_fig("07_power_users")


# ══════════════════════════════════════════════════════════════════════
# 9. SOCIAL NETWORK ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. SOCIAL NETWORK PATTERNS")
print("=" * 60)

print(f"\n── Followers Distribution ──")
print(f"  Min: {df.socialNbFollowers.min()}")
print(f"  Median: {df.socialNbFollowers.median()}")
print(f"  Mean: {df.socialNbFollowers.mean():.1f}")
print(f"  95th pctl: {df.socialNbFollowers.quantile(0.95):.0f}")
print(f"  99th pctl: {df.socialNbFollowers.quantile(0.99):.0f}")
print(f"  Max: {df.socialNbFollowers.max()}")

# Correlation between social and transactional
corr_cols = ["socialNbFollowers", "socialNbFollows", "socialProductsLiked",
             "productsListed", "productsSold", "productsBought", "productsWished"]
corr = df[corr_cols].corr()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Correlation heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=ax1, square=True, cbar_kws={"shrink": 0.8})
ax1.set_title("Correlation: Social vs Transactional", fontsize=12, fontweight="bold")
ax1.tick_params(labelsize=8)

# Followers vs products sold (for active users)
sellers = df[df["productsSold"] > 0]
ax2.scatter(sellers["socialNbFollowers"], sellers["productsSold"],
            alpha=0.4, s=15, c="#2D31FA", edgecolors="none")
ax2.set_xlabel("Followers")
ax2.set_ylabel("Products Sold")
ax2.set_title("Social Following vs Sales (sellers only)", fontsize=12, fontweight="bold")
save_fig("08_social_network")


# ══════════════════════════════════════════════════════════════════════
# 10. EXPORT CLEAN DATA FOR TABLEAU
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. EXPORT FOR TABLEAU")
print("=" * 60)

# Add computed columns for Tableau
df_export = df.copy()
df_export["segment"] = df["segment"]
df_export["is_active"] = df["is_active"]

# Map country codes to English names for Tableau geo
country_map = {
    "France": "France", "Etats-Unis": "United States", "Royaume-Uni": "United Kingdom",
    "Italie": "Italy", "Allemagne": "Germany", "Espagne": "Spain",
    "Australie": "Australia", "Danemark": "Denmark", "Suède": "Sweden",
    "Belgique": "Belgium", "Canada": "Canada", "Pays-Bas": "Netherlands",
    "Suisse": "Switzerland", "Finlande": "Finland", "Autriche": "Austria",
    "Norvège": "Norway", "Japon": "Japan", "Brésil": "Brazil",
    "Portugal": "Portugal", "Irlande": "Ireland", "Pologne": "Poland",
    "Grèce": "Greece", "Russie": "Russia", "Turquie": "Turkey",
    "Corée du Sud": "South Korea", "Chine": "China", "Inde": "India",
    "Mexique": "Mexico", "Argentine": "Argentina", "Chili": "Chile",
    "Colombie": "Colombia", "Roumanie": "Romania", "Hongrie": "Hungary",
    "République tchèque": "Czech Republic", "Lituanie": "Lithuania",
    "Lettonie": "Latvia", "Estonie": "Estonia",
}
df_export["country_en"] = df_export["country"].map(country_map).fillna(df_export["country"])

# Activity level for easier Tableau filtering
def activity_level(row):
    total = row["productsSold"] + row["productsBought"]
    if total == 0:
        return "Inactive"
    elif total <= 3:
        return "Low"
    elif total <= 10:
        return "Medium"
    else:
        return "High"

df_export["activity_level"] = df_export.apply(activity_level, axis=1)

export_path = OUTPUT_DIR / "c2c_fashion_tableau.csv"
df_export.to_csv(export_path, index=False)
print(f"\nExported {len(df_export):,} rows to {export_path}")
print(f"Columns: {list(df_export.columns)}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY — KEY FINDINGS & STORY ANGLES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — KEY FINDINGS")
print("=" * 60)

print("""
DATASET: 98,913 users from a C2C fashion marketplace (200 countries)

KEY FINDINGS:

1. MASSIVE ACTIVATION GAP
   - 98% of users have NEVER sold anything
   - 95% have NEVER bought anything
   - Only ~2-5% of registered users are transactionally active
   → Story: "The Ghost Marketplace" — understanding dormant users

2. USER SEGMENTS
   - Ghosts dominate the platform
   - Power users are a tiny but critical minority
   - Two-way traders (buy AND sell) are the rarest and most valuable

3. GEOGRAPHIC DISPARITIES
   - France dominates (25%) followed by US (21%) and UK (11%)
   - Activation rates vary significantly by country
   - Some smaller markets have disproportionately high engagement

4. GENDER DYNAMICS
   - 77% female — expected for fashion marketplace
   - Gender differences in buying vs selling behavior

5. APP = ENGAGEMENT MULTIPLIER
   - App users are significantly more active than web-only users
   - iOS users outnumber Android 4:1
   - App adoption correlates with higher transaction rates

6. SOCIAL ≠ TRANSACTIONAL
   - Social features (likes, follows) don't strongly predict transactions
   - Followers alone don't drive sales

TABLEAU DASHBOARD STORY: "Anatomy of a C2C Fashion Marketplace"
→ The activation challenge: from registration to first transaction
→ Who are the power users and what makes them different?
→ Geographic opportunities: where is engagement strongest?
→ The app effect: mobile as an activation lever
""")

print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
print("Done!")
