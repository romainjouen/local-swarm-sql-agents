import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data creation (you would replace this with your actual data query results)
data = {
    'username': ['user1', 'user1', 'user2', 'user2'],
    'number_of_likes': [10, 5, 15, 3],
    'hour_of_like': [10, 11, 10, 11],
    'item_name': ['item1', 'item2', 'item1', 'item2'],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='username', y='number_of_likes', hue='item_name')

plt.title('Likes by User and Item Over Specific Hours')
plt.xlabel('User Name')
plt.ylabel('Number of Likes')
plt.legend(title='Item Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()