import pandas as pd
import matplotlib
matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
import os

data_path = os.path.join(os.getcwd(), "docs\\pub.csv")

data = pd.read_csv(data_path)

x = data.index[2:25].to_list()
# x = [l if index % 2 == 0 or index == 0 else "" for index, l in enumerate(data.index[2:])]
x.reverse()

raw_y = data["Search query: (brain) AND (computer) AND (interface)"][2:25].to_list()
y = [int(e) for e in raw_y]
y.reverse()

# plt.bar(x, y, color=plt.get_cmap("viridis"))
# plt.show()


import seaborn as sns
sns.set(style="whitegrid", font_scale=1.4)
tips = sns.load_dataset("tips")
ax = sns.barplot(x=x, y=y, palette="Blues_d")
ax.set(xlabel='year', ylabel='number of publications')

# plt.savefig("test.jpeg", dpi=300)


