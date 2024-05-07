import json
import networkx as nx
import matplotlib.pyplot as plt

file_path = "../../assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
G = nx.DiGraph()

for item in data:
    G.add_edge(item['parent'], item['child'])

parent_values = set()
for item in data:
    parent_values.add(item['parent'])

pos = nx.spring_layout(G) 
node_size = 250 
color_list = []
for node in G.nodes():
    if node in parent_values:
        color_list.append('yellow')
    else:
        color_list.append('skyblue')
nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=3, font_weight='bold', arrows=True, arrowstyle='->', arrowsize=5, node_color= color_list)


for node in G.nodes():
    if node in parent_values:
        nx.draw_networkx_labels(G, pos, labels={node: node}, font_size=3, font_weight='bold')

parent_list = list(parent_values)
plt.annotate('Parents: ' + ' | '.join(parent_list), xy=(0.5, 1), xycoords='axes fraction', fontsize=5, ha='center', va='center')

count = 1
for node in G.nodes():
    if node in parent_values:
        count += 10
        pos[node] = (10 + count, 10 + count)

plt.savefig('relationship_graph.png', dpi=300, bbox_inches='tight')