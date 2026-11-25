import pandas as pd
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

df = pd.read_csv(url)
print("\n========== TITANIC DATASET LOADED ==========")

print("\nFirst 5 Rows:")
print(df.head())
print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Information:")
df.info()

print("\nStatistical Description:")
print(df.describe())

print("\nCategorical Data Description:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

df["age"] = df["age"].fillna(df["age"].median())

df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

df["fare"] = df["fare"].fillna(df["fare"].median())
df = df.drop(columns=["deck"])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

df = df.drop_duplicates()
print("\nDuplicates After Removing:")
print(df.duplicated().sum())

print("\nSurvival Count By Gender:")
print(df.groupby("sex")["survived"].sum())

print("\nSurvival Rate By Gender:")
print(df.groupby("sex")["survived"].mean() * 100)

print("\nSurvival Rate By Passenger Class:")
print(df.groupby("pclass")["survived"].mean() * 100)

print("\nPassenger Statistics By Gender:")

result = df.groupby("sex").agg(
    Total_Passengers=("survived", "count"),
    Average_Age=("age", "mean"),
    Average_Fare=("fare", "mean"),
    Total_Survived=("survived", "sum")
)

print(result)
print("\nGender and Class Analysis:")

group_result = df.groupby(["sex", "pclass"]).agg(
    Passengers=("survived", "count"),
    Survivors=("survived", "sum"),
    Average_Age=("age", "mean")
)

print(group_result)
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Adult", "Middle Age", "Senior"]
)

print("\nSurvival By Age Group:")
print(df.groupby("age_group", observed=True)["survived"].mean() * 100)

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Dataset:")
print(df.head())

df.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
print("\n========== PROJECT COMPLETED ==========")
print("Titanic Dataset Analysis Completed!")
print("Data Cleaning Completed!")
print("Grouping and Aggregation Completed!")
