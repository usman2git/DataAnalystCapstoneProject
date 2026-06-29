# Full Project Notebook (Parts 1–15 consolidated as a single Python script)
# You can paste this into a .py file or adapt into a Jupyter notebook.

# ============================================================
# PARTS 1–8 — FULL PROJECT FOUNDATION
# ============================================================

# -----------------------------
# PART 1 — Install & Import
# -----------------------------
!pip install pandas
!pip install matplotlib
!pip install seaborn

# =========================
# Common imports
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.max_columns", None)


# -----------------------------
# PART 2 — Load Dataset
# -----------------------------
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/n01PQ9pSmiRX6520flujwQ/survey-data.csv"
df = pd.read_csv(file_url)

print("=== PART 2: Dataset Loaded ===")
print(df.head())


# -----------------------------
# PART 3 — Basic Exploration
# -----------------------------
print("\n=== PART 3: Basic Exploration ===")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n")
print(df.describe(include="all"))


# -----------------------------
# PART 4 — Remove Duplicates
# -----------------------------
print("\n=== PART 4: Duplicate Removal ===")
dup_count = df.duplicated().sum()
print("Duplicate rows:", dup_count)

df = df.drop_duplicates()
print("Duplicates after removal:", df.duplicated().sum())


# -----------------------------
# PART 5 — Handle Missing Values
# -----------------------------
print("\n=== PART 5: Missing Value Handling ===")

categorical_cols = ["Employment", "JobSat", "RemoteWork"]
numeric_cols = ["YearsCodePro", "ConvertedCompYearly"]

# Fill categorical with mode
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Fill numeric with median
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

print("Missing values after cleaning:\n", df.isnull().sum())


# -----------------------------
# PART 6 — Basic Visualizations
# -----------------------------
print("\n=== PART 6: Basic Visualizations ===")

# Employment distribution
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Employment")
plt.xticks(rotation=45)
plt.title("Employment Distribution")
plt.show()

# Job Satisfaction distribution
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="JobSat")
plt.xticks(rotation=45)
plt.title("Job Satisfaction Distribution")
plt.show()

# YearsCodePro histogram
plt.figure(figsize=(10,5))
sns.histplot(df["YearsCodePro"], kde=True)
plt.title("Distribution of Professional Coding Experience")
plt.show()


# -----------------------------
# PART 7 — Removing Duplicates (Second Dataset)
# -----------------------------
print("\n=== PART 7: Duplicate Handling (survey-data-duplicates.csv) ===")

file_path_dup = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UDKAZw-kz18Yj8P6icf_qw/survey-data-duplicates.csv"
df7 = pd.read_csv(file_path_dup)

print(df7.head())

# Task 1: Identify duplicate rows
dup_count7 = df7.duplicated().sum()
print("Duplicate rows:", dup_count7)

print("Sample duplicate rows:")
print(df7[df7.duplicated()].head())

# Task 2: Remove duplicates
df7 = df7.drop_duplicates()
print("Duplicates after removal:", df7.duplicated().sum())

# Task 3: Missing values
print("Missing values:\n", df7.isnull().sum())

# Impute EdLevel with mode
if "EdLevel" in df7.columns:
    df7["EdLevel"] = df7["EdLevel"].fillna(df7["EdLevel"].mode()[0])

# Normalize compensation
if "ConvertedCompYearly" in df7.columns:
    df7["ConvertedCompYearly"] = df7["ConvertedCompYearly"].fillna(df7["ConvertedCompYearly"].median())

print("Missing values after imputation:\n", df7.isnull().sum())


# -----------------------------
# PART 8 — Finding Missing Values (Heatmap + Imputation)
# -----------------------------
print("\n=== PART 8: Missing Value Analysis ===")

file_path_dup2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UDKAZw-kz18Yj8P6icf_qw/survey-data-duplicates.csv"
df8 = pd.read_csv(file_path_dup2)

# Task 1: Basic info
print(df8.info())
print(df8.describe(include="all"))

# Task 2: Missing values per column
print("Missing values:\n", df8.isnull().sum())

# Task 3: Heatmap of missing values
plt.figure(figsize=(12,6))
sns.heatmap(df8.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Task 4: Missing rows in Employment
missing_emp = df8["Employment"].isnull().sum()
print("Missing Employment rows:", missing_emp)

# Task 5: Most frequent Employment value
most_freq_emp = df8["Employment"].mode()[0]
print("Most frequent Employment:", most_freq_emp)

# Task 6: Impute missing Employment
df8["Employment"] = df8["Employment"].fillna(most_freq_emp)
print("Missing Employment after imputation:", df8["Employment"].isnull().sum())

# Task 7: Visualize Employment distribution
plt.figure(figsize=(10,5))
df8["Employment"].value_counts().plot(kind="bar")
plt.title("Employment Distribution After Imputation")
plt.xlabel("Employment Type")
plt.ylabel("Count")
plt.show()


pd.set_option("display.max_columns", None)

# =============================================================================
# PART 9 – Impute Missing Values (survey-data-duplicates.csv)
# =============================================================================

print("\n=== PART 9: Impute Missing Values ===")

file_path_dup = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UDKAZw-kz18Yj8P6icf_qw/survey-data-duplicates.csv"
df_dup = pd.read_csv(file_path_dup)
print(df_dup.head())

# Task 1: Identify duplicate rows
dup_mask = df_dup.duplicated()
num_duplicates = dup_mask.sum()
print("Number of duplicate rows:", num_duplicates)
duplicate_rows = df_dup[dup_mask]
print("Duplicate rows:\n", duplicate_rows)

# Task 2: Remove duplicate rows
df_dup = df_dup.drop_duplicates()
print("Shape after removing duplicates:", df_dup.shape)

# Task 3: Find missing values for all columns
missing_all = df_dup.isnull().sum()
print("Missing values per column:\n", missing_all)

# Task 4: How many rows are missing in RemoteWork
missing_remote = df_dup["RemoteWork"].isnull().sum()
print("Missing rows in RemoteWork:", missing_remote)

# Task 5: Value counts for RemoteWork
remote_counts = df_dup["RemoteWork"].value_counts(dropna=False)
print("RemoteWork value counts:\n", remote_counts)

# Task 6: Identify majority value in RemoteWork
majority_remote = df_dup["RemoteWork"].mode()[0]
print("Majority value in RemoteWork:", majority_remote)

# Task 7: Impute empty rows in RemoteWork with majority value
df_dup["RemoteWork"] = df_dup["RemoteWork"].fillna(majority_remote)
print("Missing RemoteWork after imputation:", df_dup["RemoteWork"].isnull().sum())

# Task 8: Check compensation-related columns and describe distribution
comp_cols = [c for c in df_dup.columns if "Comp" in c]
print("Compensation-related columns:", comp_cols)
print(df_dup[comp_cols].describe())

# =============================================================================
# PART 10 – Data Normalization Techniques (survey-data-duplicates.csv)
# =============================================================================

print("\n=== PART 10: Data Normalization Techniques ===")

df_norm = pd.read_csv(file_path_dup)
print(df_norm.head())

# Section 1: Handling duplicates
num_dup_norm = df_norm.duplicated().sum()
print("Duplicates before:", num_dup_norm)
df_norm = df_norm.drop_duplicates()
print("Duplicates after:", df_norm.duplicated().sum())

# Section 2: Missing values in CodingActivities
missing_coding = df_norm["CodingActivities"].isnull().sum()
print("Missing in CodingActivities:", missing_coding)

# Impute missing values in CodingActivities with forward-fill
df_norm["CodingActivities"] = df_norm["CodingActivities"].fillna(method="ffill")
print("Missing in CodingActivities after ffill:", df_norm["CodingActivities"].isnull().sum())

# Section 3: Normalizing compensation data
# Identify compensation-related columns
comp_cols_norm = [c for c in df_norm.columns if "ConvertedCompYearly" in c or "CompTotal" in c]
print("Compensation-related columns:", comp_cols_norm)

# Handle NaN in ConvertedCompYearly (drop rows with NaN for normalization)
df_norm_comp = df_norm.dropna(subset=["ConvertedCompYearly"]).copy()

# Task 5: Min-Max scaling
col = "ConvertedCompYearly"
min_val = df_norm_comp[col].min()
max_val = df_norm_comp[col].max()
df_norm_comp[col + "_MinMax"] = (df_norm_comp[col] - min_val) / (max_val - min_val)

# Task 6: Z-score normalization
mean_val = df_norm_comp[col].mean()
std_val = df_norm_comp[col].std()
df_norm_comp[col + "_Zscore"] = (df_norm_comp[col] - mean_val) / std_val

# Task 7: Visualize distributions
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(df_norm_comp[col], bins=30)
plt.title("Original ConvertedCompYearly")

plt.subplot(1, 3, 2)
plt.hist(df_norm_comp[col + "_MinMax"], bins=30)
plt.title("Min-Max Scaled")

plt.subplot(1, 3, 3)
plt.hist(df_norm_comp[col + "_Zscore"], bins=30)
plt.title("Z-score Scaled")

plt.tight_layout()
plt.show()

# =============================================================================
# PART 11 – Data Wrangling Lab (survey-data.csv)
# =============================================================================

print("\n=== PART 11: Data Wrangling Lab ===")

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/n01PQ9pSmiRX6520flujwQ/survey-data.csv"
df = pd.read_csv(dataset_url)
print(df.head())

# 2.1 Summarize dataset: dtypes, counts, missing
print(df.info())
print("Missing values per column:\n", df.isnull().sum())

# 2.2 Basic statistics for numerical columns
print(df.describe())

# 3.1 Identify inconsistent/irrelevant entries in Country
print("Unique Country values (sample):", df["Country"].unique()[:20])

# Example: remove rows where Country is NaN or 'Other'
df = df[~df["Country"].isnull()]
df = df[df["Country"] != "Other"]

# 3.2 Standardize Country or EdLevel (simple mapping example)
country_map = {
    "United States of America": "USA",
    "United Kingdom": "UK",
}
df["Country"] = df["Country"].replace(country_map)

# 4.1 One-hot encode Employment
employment_dummies = pd.get_dummies(df["Employment"], prefix="Employment")
df = pd.concat([df, employment_dummies], axis=1)

# 5.1 Columns with highest missing values
missing_sorted = df.isnull().sum().sort_values(ascending=False)
print("Columns with highest missing:\n", missing_sorted.head(10))

# 5.2 Impute numerical column ConvertedCompYearly with median
if "ConvertedCompYearly" in df.columns:
    median_comp = df["ConvertedCompYearly"].median()
    df["ConvertedCompYearly"] = df["ConvertedCompYearly"].fillna(median_comp)

# 5.3 Impute categorical column RemoteWork with mode
if "RemoteWork" in df.columns:
    mode_remote = df["RemoteWork"].mode()[0]
    df["RemoteWork"] = df["RemoteWork"].fillna(mode_remote)

# 6.1 Min-Max scaling ConvertedCompYearly
if "ConvertedCompYearly" in df.columns:
    comp_min = df["ConvertedCompYearly"].min()
    comp_max = df["ConvertedCompYearly"].max()
    df["ConvertedCompYearly_MinMax"] = (df["ConvertedCompYearly"] - comp_min) / (comp_max - comp_min)

# 6.2 Log-transform ConvertedCompYearly
df["ConvertedCompYearly_Log"] = np.log1p(df["ConvertedCompYearly"])

# 7.1 Feature engineering: ExperienceLevel from YearsCodePro
def experience_level(years):
    try:
        y = float(years)
    except:
        return np.nan
    if y < 3:
        return "Beginner"
    elif y < 7:
        return "Intermediate"
    elif y < 15:
        return "Experienced"
    else:
        return "Expert"

if "YearsCodePro" in df.columns:
    df["ExperienceLevel"] = df["YearsCodePro"].apply(experience_level)

# =============================================================================
# PART 12 – Lab: Exploratory Data Analysis (survey-data.csv)
# =============================================================================

print("\n=== PART 12: Exploratory Data Analysis Lab ===")

df_eda = pd.read_csv(dataset_url)

# Step 3: Handling missing data in Employment, JobSat, RemoteWork
for col in ["Employment", "JobSat", "RemoteWork"]:
    if col in df_eda.columns:
        mode_val = df_eda[col].mode()[0]
        df_eda[col] = df_eda[col].fillna(mode_val)

# Step 4: Experience vs JobSat
if "YearsCodePro" in df_eda.columns and "JobSatPoints_1" in df_eda.columns:
    # Convert YearsCodePro to numeric
    def to_float(x):
        try:
            return float(x)
        except:
            return np.nan

    df_eda["YearsCodePro_num"] = df_eda["YearsCodePro"].apply(to_float)

    bins = [0, 5, 10, 20, np.inf]
    labels = ["0-5", "5-10", "10-20", ">20"]
    df_eda["ExperienceRange"] = pd.cut(df_eda["YearsCodePro_num"], bins=bins, labels=labels)

    median_job_sat = df_eda.groupby("ExperienceRange")["JobSatPoints_1"].median()
    print("Median JobSatPoints_1 by ExperienceRange:\n", median_job_sat)

    median_job_sat.plot(kind="bar")
    plt.title("Median Job Satisfaction by Experience Range")
    plt.ylabel("Median JobSatPoints_1")
    plt.show()

# Step 5: Visualize JobSat distribution
if "JobSat" in df_eda.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_eda, x="JobSat")
    plt.xticks(rotation=45)
    plt.title("JobSat Distribution")
    plt.tight_layout()
    plt.show()

# Step 6: RemoteWork preferences by Employment
if "RemoteWork" in df_eda.columns and "Employment" in df_eda.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_eda, x="RemoteWork", hue="Employment")
    plt.xticks(rotation=45)
    plt.title("Remote Work by Employment Type")
    plt.tight_layout()
    plt.show()

# Step 7: Programming language trends by region (simple example)
if "LanguageHaveWorkedWith" in df_eda.columns and "Country" in df_eda.columns:
    # Example: filter for USA
    usa = df_eda[df_eda["Country"] == "United States of America"]
    lang_series = usa["LanguageHaveWorkedWith"].dropna().str.split(";")
    lang_flat = [item for sublist in lang_series for item in sublist]
    lang_counts = pd.Series(lang_flat).value_counts().head(10)
    lang_counts.plot(kind="bar")
    plt.title("Top Languages in USA (Have Worked With)")
    plt.ylabel("Count")
    plt.show()

# Step 8: Correlation between YearsCodePro and JobSatPoints_1
if "YearsCodePro_num" in df_eda.columns and "JobSatPoints_1" in df_eda.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df_eda, x="YearsCodePro_num", y="JobSatPoints_1")
    plt.title("Experience vs Job Satisfaction")
    plt.show()
    corr_val = df_eda[["YearsCodePro_num", "JobSatPoints_1"]].corr().iloc[0, 1]
    print("Correlation (YearsCodePro_num vs JobSatPoints_1):", corr_val)

# Step 9: EdLevel vs Employment
if "EdLevel" in df_eda.columns and "Employment" in df_eda.columns:
    ct = pd.crosstab(df_eda["EdLevel"], df_eda["Employment"])
    print("Cross-tab EdLevel vs Employment:\n", ct)
    ct.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Employment by Education Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Step 10: Save cleaned/analyzed dataset
df_eda.to_csv("survey_data_eda_cleaned.csv", index=False)

# =============================================================================
# PART 13 – Finding How The Data Is Distributed (survey-data.csv)
# =============================================================================

print("\n=== PART 13: Finding How The Data Is Distributed ===")

df_dist = pd.read_csv(dataset_url)

# Step 2: Examine structure
print(df_dist.info())
print(df_dist.describe())

# Step 3: Handle missing data (simple fill for key columns)
for col in ["Employment", "JobSat", "YearsCodePro"]:
    if col in df_dist.columns:
        df_dist[col] = df_dist[col].fillna(df_dist[col].mode()[0])

# Step 4: Analyze key columns
for col in ["Employment", "JobSat", "YearsCodePro"]:
    if col in df_dist.columns:
        print(f"Value counts for {col}:\n", df_dist[col].value_counts())

# Step 5: Visualize JobSat
if "JobSat" in df_dist.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_dist, x="JobSat")
    plt.xticks(rotation=45)
    plt.title("JobSat Distribution")
    plt.tight_layout()
    plt.show()

# Step 6: Programming languages analysis (Have vs Want)
if "LanguageHaveWorkedWith" in df_dist.columns and "LanguageWantToWorkWith" in df_dist.columns:
    have = df_dist["LanguageHaveWorkedWith"].dropna().str.split(";")
    want = df_dist["LanguageWantToWorkWith"].dropna().str.split(";")
    have_flat = [x for sub in have for x in sub]
    want_flat = [x for sub in want for x in sub]
    have_counts = pd.Series(have_flat).value_counts().head(10)
    want_counts = pd.Series(want_flat).value_counts().head(10)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    have_counts.plot(kind="bar")
    plt.title("Top Languages (Have Worked With)")
    plt.subplot(1, 2, 2)
    want_counts.plot(kind="bar")
    plt.title("Top Languages (Want To Work With)")
    plt.tight_layout()
    plt.show()

# Step 7: RemoteWork by region
if "RemoteWork" in df_dist.columns and "Country" in df_dist.columns:
    ct_rw = pd.crosstab(df_dist["Country"], df_dist["RemoteWork"])
    ct_rw.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Remote Work by Country")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Step 8: Correlation JobSat vs YearsCodePro
if "YearsCodePro" in df_dist.columns and "JobSatPoints_1" in df_dist.columns:
    df_dist["YearsCodePro_num"] = df_dist["YearsCodePro"].apply(to_float)
    corr_j = df_dist[["YearsCodePro_num", "JobSatPoints_1"]].corr().iloc[0, 1]
    print("Correlation (YearsCodePro_num vs JobSatPoints_1):", corr_j)

# Step 9: Employment vs EdLevel
if "EdLevel" in df_dist.columns and "Employment" in df_dist.columns:
    ct2 = pd.crosstab(df_dist["EdLevel"], df_dist["Employment"])
    ct2.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Employment vs Education Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Step 10: Export cleaned data
df_dist.to_csv("survey_data_distribution_cleaned.csv", index=False)

# =============================================================================
# PART 14 – Finding Outliers (survey-data.csv)
# =============================================================================

print("\n=== PART 14: Finding Outliers ===")

df_out = pd.read_csv(dataset_url)

# Step 2: Distribution of Industry
if "Industry" in df_out.columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df_out, x="Industry")
    plt.xticks(rotation=90)
    plt.title("Distribution by Industry")
    plt.tight_layout()
    plt.show()

# Step 3: High compensation outliers
if "ConvertedCompYearly" in df_out.columns:
    comp = df_out["ConvertedCompYearly"].dropna()
    mean_c = comp.mean()
    median_c = comp.median()
    std_c = comp.std()
    print("ConvertedCompYearly stats: mean=", mean_c, "median=", median_c, "std=", std_c)

    threshold = mean_c + 3 * std_c
    high_outliers = df_out[df_out["ConvertedCompYearly"] > threshold]
    print("Number of high compensation outliers:", high_outliers.shape[0])

# Step 4: Detect outliers via IQR
if "ConvertedCompYearly" in df_out.columns:
    Q1 = comp.quantile(0.25)
    Q3 = comp.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df_out["ConvertedCompYearly"] < lower_bound) | (df_out["ConvertedCompYearly"] > upper_bound)
    print("Number of IQR outliers:", outlier_mask.sum())

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_out["ConvertedCompYearly"])
    plt.title("Boxplot of ConvertedCompYearly (with outliers)")
    plt.show()

# Step 5: Remove outliers and create new DataFrame
df_no_out = df_out[~outlier_mask].copy()
print("Original shape:", df_out.shape, "Shape without outliers:", df_no_out.shape)

# Step 6: Correlation analysis with Age mapped to numeric
if "Age" in df_no_out.columns:
    age_map = {
        "Under 18 years old": 16,
        "18-24 years old": 21,
        "25-34 years old": 29,
        "35-44 years old": 39,
        "45-54 years old": 49,
        "55-64 years old": 59,
        "65 years or older": 70,
    }
    df_no_out["Age_num"] = df_no_out["Age"].map(age_map)

    num_cols = df_no_out.select_dtypes(include=[np.number])
    corr_matrix = num_cols.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix (No Outliers)")
    plt.show()

# =============================================================================
# PART 15 – Finding Correlation (survey-data.csv)
# =============================================================================

print("\n=== PART 15: Finding Correlation ===")

df_corr = pd.read_csv(dataset_url)

# Step 3: Distribution of ConvertedCompYearly
if "ConvertedCompYearly" in df_corr.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_corr["ConvertedCompYearly"].dropna(), bins=30, kde=True)
    plt.title("Distribution of ConvertedCompYearly")
    plt.tight_layout()
    plt.show()

# Step 4: Median compensation for full-time employees
if "Employment" in df_corr.columns and "ConvertedCompYearly" in df_corr.columns:
    full_time = df_corr[df_corr["Employment"] == "Employed, full-time"]
    median_full_time = full_time["ConvertedCompYearly"].median()
    print("Median ConvertedCompYearly (full-time):", median_full_time)

# Step 5: Compensation range by Country (boxplots)
if "Country" in df_corr.columns and "ConvertedCompYearly" in df_corr.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_corr, x="Country", y="ConvertedCompYearly")
    plt.xticks(rotation=90)
    plt.title("ConvertedCompYearly by Country")
    plt.tight_layout()
    plt.show()

# Step 6: Remove outliers from ConvertedCompYearly (IQR)
comp2 = df_corr["ConvertedCompYearly"].dropna()
Q1_2 = comp2.quantile(0.25)
Q3_2 = comp2.quantile(0.75)
IQR_2 = Q3_2 - Q1_2
lower_2 = Q1_2 - 1.5 * IQR_2
upper_2 = Q3_2 + 1.5 * IQR_2
mask2 = (df_corr["ConvertedCompYearly"] >= lower_2) & (df_corr["ConvertedCompYearly"] <= upper_2)
df_corr_clean = df_corr[mask2].copy()
print("Shape after removing outliers:", df_corr_clean.shape)

# Step 7: Correlations between ConvertedCompYearly, WorkExp, JobSatPoints_1
for col in ["WorkExp", "JobSatPoints_1"]:
    if col in df_corr_clean.columns:
        df_corr_clean[col] = pd.to_numeric(df_corr_clean[col], errors="coerce")

corr_subset = df_corr_clean[["ConvertedCompYearly", "WorkExp", "JobSatPoints_1"]].corr()
print("Correlation matrix:\n", corr_subset)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_subset, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation: Compensation, WorkExp, JobSatPoints_1")
plt.tight_layout()
plt.show()

# Step 8: Scatter plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_corr_clean, x="WorkExp", y="ConvertedCompYearly")
plt.title("ConvertedCompYearly vs WorkExp")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_corr_clean, x="JobSatPoints_1", y="ConvertedCompYearly")
plt.title("ConvertedCompYearly vs JobSatPoints_1")

plt.tight_layout()
plt.show()

print("\n=== All parts executed. You can adapt this script into a notebook as needed. ===")
