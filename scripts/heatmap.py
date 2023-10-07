import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.rand(10, 10)

# Create heatmap
fig, ax = plt.subplots()
heatmap = ax.imshow(data)
print("heatmap", heatmap)

print("data", data)
# Customize the heatmap
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.colorbar(heatmap)

# Display the heatmap
plt.show()
plt.savefig('heatmap1.png')