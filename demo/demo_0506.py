import pandas as pd
import matplotlib.pyplot as plt

# Data from SQL query result
data = [
    {"username": "alice_ai", "hour_of_like": 10, "item_name": "New breakthrough in reinforcement learning"},
    {"username": "bob_ml", "hour_of_like": 15, "item_name": "GPT-4 shows impressive results in medical diagnosis"},
    {"username": "charlie_nlp", "hour_of_like": 17, "item_name": "L'IA générative révolutionne la création artistique"}
]

# Create a DataFrame
df = pd.DataFrame(data)

# Generate the bar chart
plt.figure(figsize=(12, 6))

for item in df['item_name'].unique():
    subset = df[df['item_name'] == item]
    plt.bar(subset['username'], subset['hour_of_like'], label=item)

plt.xlabel('Username')
plt.ylabel('Hour of Like')
plt.title('Likes Between "2023-04-13" and "2023-04-15 11:00:00", Colored by Item Name')
plt.legend(title='Item Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()