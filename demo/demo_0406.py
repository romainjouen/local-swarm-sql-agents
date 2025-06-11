import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'username': ['alice_ai', 'bob_ml', 'charlie_nlp'],
    'like_hour': [10, 15, 17],
    'title': [
        'New breakthrough in reinforcement learning',
        'GPT-4 shows impressive results in medical diagnosis',
        "L'IA générative révolutionne la création artistique"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(8, 5))
for item in df['title'].unique():
    plt.bar(df[df['title'] == item]['username'], 
            df[df['title'] == item]['like_hour'], 
            label=item)

plt.title('Hour of Likes by User')
plt.xlabel('User Name')
plt.ylabel('Hour of Likes')
plt.legend()
plt.show()