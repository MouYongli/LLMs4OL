import pandas as pd
import matplotlib.pyplot as plt

# 原始数据
geonames_data = {
    "Dataset": ["GeoNames"]*13,
    "Model": [
        "BERT-Large", "PubMedBERT", "BART-Large", "Flan-T5-Large", "BLOOM-1b7",
        "Flan-T5-XL", "BLOOM-3b", "LLaMA-7B", "GPT-3", "GPT-3.5", "GPT-4",
        "Flan-T5-Large*", "Flan-T5-XL*"
    ],
    "$t_1$": [41.005, None, 38.114, 59.635, 33.169, 49.372, 35.856, 33.496, 43.431, 59.405, 38.561, 42.539, 48.414],
    "$t_2$": [51.698, None, 41.033, 48.246, 31.049, 44.057, 39.120, 33.496, 51.742, 47.792, 52.465, 59.403, 34.803],
    "$t_3$": [40.557, None, 40.552, 54.082, 33.169, 45.098, 53.922, 33.496, 42.700, 67.782, 34.004, 40.299, 55.231],
    "$t_4$": [48.703, None, 52.500, 48.241, 32.839, 52.413, 30.227, 33.496, 53.202, 41.951, 38.897, 62.466, 46.964],
    "$t_5$": [37.165, None, 39.094, 44.404, 33.775, 43.929, 35.627, 33.496, 46.040, 48.026, 44.069, 46.034, 57.484],
    "$t_6$": [41.070, None, 45.801, 51.309, 33.530, 46.348, 33.606, 33.496, 52.566, 51.728, 55.433, 57.415, 36.293],
    "$t_7$": [41.707, None, 36.671, 36.407, 36.674, 49.982, 48.263, 33.496, 45.496, 45.257, 33.782, 42.496, 59.057],
    "$t_8$": [54.547, None, 55.400, 38.449, 32.922, 44.298, 37.731, 33.496, 52.626, 43.860, 36.234, 62.045, 49.261]
}
umls_data = {
    "Dataset": ["UMLS"]*13,
    "Model": [
        "BERT-Large", "PubMedBERT", "BART-Large", "Flan-T5-Large", "BLOOM-1b7",
        "Flan-T5-XL", "BLOOM-3b", "LLaMA-7B", "GPT-3", "GPT-3.5", "GPT-4",
        "Flan-T5-Large*", "Flan-T5-XL*"
    ],
    "$t_1$": [48.215, 33.713, 36.029, 47.558, 33.713, 64.256, 33.169, 32.948, 51.584, 61.380, 41.195, 37.176, 63.693],
    "$t_2$": [38.842, 33.713, 48.218, 51.221, 36.188, 46.533, 37.233, 32.948, 49.412, 70.385, 76.999, 48.667, 50.046],
    "$t_3$": [41.467, 33.713, 41.429, 55.320, 33.713, 51.006, 34.823, 32.948, 49.865, 63.915, 42.558, 36.074, 36.917],
    "$t_4$": [40.412, 33.713, 49.907, 40.947, 38.262, 41.549, 35.777, 32.948, 42.901, 66.821, 63.889, 42.121, 41.343],
    "$t_5$": [45.889, 33.713, 39.372, 49.455, 33.713, 60.077, 33.169, 32.948, 50.573, 63.144, 50.288, 48.396, 78.127],
    "$t_6$": [40.911, 33.713, 47.479, 50.873, 35.895, 42.831, 35.895, 32.948, 46.070, 67.271, 78.116, 46.654, 50.122],
    "$t_7$": [41.041, 33.713, 42.398, 44.232, 33.278, 51.257, 33.059, 32.948, 45.367, 56.648, 36.594, 53.428, 79.255],
    "$t_8$": [42.922, 33.713, 45.464, 42.909, 33.605, 41.186, 37.483, 32.948, 46.728, 64.412, 60.728, 35.970, 39.274]
}

schema_org_data = {
    "Dataset": ["schema.org"]*13,
    "Model": [
        "BERT-Large", "PubMedBERT", "BART-Large", "Flan-T5-Large", "BLOOM-1b7",
        "Flan-T5-XL", "BLOOM-3b", "LLaMA-7B", "GPT-3", "GPT-3.5", "GPT-4",
        "Flan-T5-Large*", "Flan-T5-XL*"
    ],
    "$t_1$": [43.851, None, 34.628, 46.983, 33.395, 42.708, 41.643, 33.374, 49.646, 56.843, 58.479, 35.358, 91.063],
    "$t_2$": [41.172, None, 38.693, 49.924, 47.833, 33.455, 47.169, 33.374, 49.289, 74.385, 72.827, 85.436, 57.469],
    "$t_3$": [44.067, None, 39.281, 46.118, 33.395, 33.591, 47.980, 33.374, 50.977, 58.525, 65.831, 29.824, 74.688],
    "$t_4$": [43.200, None, 52.909, 54.788, 39.777, 42.766, 45.255, 33.374, 48.031, 70.164, 63.306, 89.248, 65.329],
    "$t_5$": [43.703, None, 38.203, 40.277, 38.925, 36.694, 39.733, 33.374, 47.191, 53.359, 50.565, 41.305, 91.541],
    "$t_6$": [40.054, None, 41.170, 54.479, 48.568, 34.041, 40.758, 33.374, 48.632, 72.354, 74.247, 91.681, 50.635],
    "$t_7$": [42.151, None, 43.261, 42.060, 44.357, 33.751, 51.280, 33.374, 48.878, 54.165, 57.452, 42.461, 91.709],
    "$t_8$": [43.720, None, 42.744, 47.930, 39.578, 36.456, 48.736, 33.374, 49.489, 71.030, 63.694, 56.395, 33.333]
}

# 创建DataFrame
df1 = pd.DataFrame(geonames_data)
df2 = pd.DataFrame(umls_data)
df3 = pd.DataFrame(schema_org_data)

# 绘制图表1
plt.figure(figsize=(10, 6))
bar_width = 0.1
positions = range(len(df1["Model"]))
for i in range(1, 9):
    plt.bar([pos + i*bar_width for pos in positions], df1[f"$t_{i}$"], width=bar_width, label=f"$t_{i}$")
plt.xticks(positions, df1["Model"], rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('Dataset GeoNames F1 Score For Different Models')
plt.legend()
plt.tight_layout()
plt.show()

# 绘制图表2
plt.figure(figsize=(10, 6))
bar_width = 0.1
positions = range(len(df2["Model"]))
for i in range(1, 9):
    plt.bar([pos + i*bar_width for pos in positions], df2[f"$t_{i}$"], width=bar_width, label=f"$t_{i}$")
plt.xticks(positions, df2["Model"], rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('Dataset UMLS F1 Score For Different Models')
plt.legend()
plt.tight_layout()
plt.show()

# 绘制图表3
plt.figure(figsize=(10, 6))
bar_width = 0.1
positions = range(len(df3["Model"]))
for i in range(1, 9):
    plt.bar([pos + i*bar_width for pos in positions], df3[f"$t_{i}$"], width=bar_width, label=f"$t_{i}$")
plt.xticks(positions, df3["Model"], rotation=45, ha='right')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('Dataset schema.org F1 Score For Different Models')
plt.legend()
plt.tight_layout()
plt.show()