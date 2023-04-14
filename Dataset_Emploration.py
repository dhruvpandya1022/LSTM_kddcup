# Dataset Exploration
# data1 = datasets.fetch_kddcup99(subset='SA')
# df = pd.DataFrame(data1.data, columns=num_features)

print("Start Plotting")
# fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))
# for i, ax in enumerate(axes.flatten()):
#     if i < len(num_features):
#         print('1st Iteration')
#         sns.histplot(data[num_features[i]], ax=ax)
#         ax.set_title(num_features[i])
#     else:
#         ax.axis('off')

# Plot the correlation matrix
# Load the data

data1 = datasets.fetch_kddcup99(subset='SA')

# Convert data to pandas dataframe
df = pd.DataFrame(data1.data, columns=data1.feature_names)

corr = df.corr()
print(corr)
# sns.heatmap(corr, annot=True, cmap='coolwarm')

# Plot the number of instances for each class
class_counts = pd.DataFrame(data1.target, columns=['class']).groupby('class').size()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Number of Instances for Each Class')
plt.xlabel('Class')
plt.ylabel('Number of Instances')

plt.show()
