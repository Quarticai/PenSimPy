import pandas as pd
import matplotlib.pyplot as plt

good_dfs = pd.read_csv('good_dfs.csv', index_col=None)
print(good_dfs)
bad_dfs = pd.read_csv('bad_df.csv', index_col=None)
print(bad_dfs)
base_df = pd.read_csv('base_df.csv', index_col=None)
print(base_df)

plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('base')
plt.hist(base_df['Temperature'], bins=100)
plt.subplot(2, 2, 2)
plt.title('good')
plt.hist(good_dfs['Temperature'].head(1150), bins=100)
plt.subplot(2, 2, 3)
plt.title('bad')
plt.hist(bad_dfs['Temperature'], bins=100)
plt.subplot(2, 2, 4)
plt.title('all')
plt.hist(bad_dfs['Temperature'], bins=100, alpha=0.5, label='bad', color='b')
plt.hist(base_df['Temperature'], bins=100, alpha=0.5, label='base', color='r')
plt.hist(good_dfs['Temperature'].head(1150), bins=100, alpha=0.5, label='good', color='g')
plt.legend()
plt.tight_layout()

plt.figure(2)
plt.title('all')
plt.hist(bad_dfs['Base flow rate'], bins=100, alpha=0.5, label='bad', color='b')
plt.hist(base_df['Base flow rate'], bins=100, alpha=0.5, label='base', color='r')
plt.hist(good_dfs['Base flow rate'], bins=100, alpha=0.5, label='good', color='g')
plt.legend()
plt.tight_layout()

plt.figure(3)
plt.title('all')
batch_len = 1150
num_batch = 7
feature = 'Base flow rate'
plt.hist(bad_dfs[feature], bins=100, alpha=0.5, label='bad', color='b')
plt.hist(base_df[feature], bins=100, alpha=0.5, label='base', color='r')
for i in range(1, 11):
    plt.hist(good_dfs[feature].iloc[batch_len * (i - 1): batch_len * i], bins=100, alpha=0.5,
             label='good', color='g')
plt.legend()
plt.tight_layout()
plt.show()
