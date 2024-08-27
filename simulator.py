import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

data_species = pd.read_csv('especies.csv')
data_interactions = pd.read_csv('predacion.csv')

G = nx.DiGraph()
for _, row in data_species.iterrows():
    G.add_node(row['Nombre_Común'], 
               Cantidad=row['Cantidad_Inicial'],
               humedad=row['Humedad_Óptima (%)'], 
               temperatura=row['Temperatura_Óptima (°C)'])

for _, row in data_interactions.iterrows():
    depredador = row['Depredador']
    presas = row['Presas'].split(',')
    for presa in presas:
        G.add_edge(depredador.strip(), presa.strip())

pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, width=0.5, alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='darkgrey')

labels = {node: f"{node}\nCantidad: {G.nodes[node]['Cantidad']}\nHumedad: {G.nodes[node]['humedad']}%\nTemp. Óptima: {G.nodes[node]['temperatura']}°C" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
plt.title('Grafo de Interacciones en un Ecosistema', fontsize=15)
plt.axis('off')

st.write("""
# Grafo de Interacciones en un Ecosistema
""")
st.pyplot(plt)

#Determinar el grado de los nodos, que como es el caso de un grafo dirigido, tiene in degree y out degree, lo que nos ayuda a determinar las especies con mas interaccoiones
#en la red, quienes son los mayores depredadores que serian las de mayor out-degree y las especies más vulnerables que serian las de mayor in-degree
in_degrees = G.in_degree()
out_degrees = G.out_degree()
in_degree_df = pd.DataFrame(in_degrees, columns=['Nodo', 'Grado de Entrada'])
out_degree_df = pd.DataFrame(out_degrees, columns=['Nodo', 'Grado de Salida'])
out_degree_df = out_degree_df.drop('Nodo', axis=1)

degree = pd.concat([in_degree_df, out_degree_df], axis=1)
degree["Degree"]=degree["Grado de Entrada"]+degree["Grado de Salida"]
st.write("""## Grado de los nodos""")
st.dataframe(degree)

#Centralidad del grafo teniendo en cuenta el grado
degree_centrality = nx.degree_centrality(G)
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_color='red', node_size=2000, font_size=12, ax=ax)
labels = {node: f"{node}\n{centrality:.2f}" for node, centrality in degree_centrality.items()}
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)

st.pyplot(fig) 

st.write("Centralidad de grado:")
st.write(degree_centrality)



