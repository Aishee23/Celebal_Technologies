import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
df = sns.load_dataset('iris')

# Pair plot
sns.pairplot(df, hue='species')
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# Histograms
df.hist(bins=20, figsize=(10, 8))
plt.suptitle('Histograms of Iris Features')
plt.show()

# Density plots
df.plot(kind='density', subplots=True, layout=(2, 2), sharex=False, figsize=(10, 8))
plt.suptitle('Density Plots of Iris Features')
plt.show()

# Box plots
df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False, figsize=(10, 8))
plt.suptitle('Box Plots of Iris Features')
plt.show()

# Box plots with species
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient='h', palette='Set3')
plt.title('Box Plots of Iris Features by Species')
plt.show()

# Violin plots
plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='sepal_length', data=df, palette='Set3')
plt.title('Violin Plot of Sepal Length by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='sepal_width', data=df, palette='Set3')
plt.title('Violin Plot of Sepal Width by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='petal_length', data=df, palette='Set3')
plt.title('Violin Plot of Petal Length by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='petal_width', data=df, palette='Set3')
plt.title('Violin Plot of Petal Width by Species')
plt.show()

# Heatmap of correlations
corr = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Feature Correlations')
plt.show()

# Scatter plots
plt.figure(figsize=(10, 8))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df, palette='Set2')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df, palette='Set2')
plt.title('Scatter Plot of Petal Length vs Petal Width')
plt.show()
