import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib  
matplotlib.use("TKAGG")
from scipy.integrate import odeint
import numpy as np

def load_data(species_file, interactions_file):
    data_species = pd.read_csv(species_file)
    data_interactions = pd.read_csv(interactions_file)
    return data_species, data_interactions

def create_interaction_graph(data_species, data_interactions):
    G = nx.DiGraph()
    
    for _, row in data_species.iterrows():
        G.add_node(row['Nombre_Común'], 
                   Cantidad=row['Cantidad_Inicial'],
                   Tasa_Natalidad=row['Tasa_Natalidad'],
                   Tasa_Mortalidad=row['Tasa_Mortalidad'])
    
    for _, row in data_interactions.iterrows():
        depredador = row['Depredador']
        presas = row['Presas'].split(',')
        for presa in presas:
            G.add_edge(depredador.strip(), presa.strip(), tasa=row['Tasa de predación'])
    
    return G

def plot_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, width=0.5, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='darkgrey')

    labels = {node: f"{node}\nCantidad: {G.nodes[node]['Cantidad']}\nTasa Natalidad: {G.nodes[node]['Tasa_Natalidad']}\nTasa Mortalidad: {G.nodes[node]['Tasa_Mortalidad']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
    plt.title('Grafo de Interacciones en un Ecosistema', fontsize=15)
    plt.axis('off')
    st.pyplot(plt)

def calculate_degrees(G):
    in_degrees = G.in_degree()
    out_degrees = G.out_degree()
    
    in_degree_df = pd.DataFrame(in_degrees, columns=['Nodo', 'Grado de Entrada'])
    out_degree_df = pd.DataFrame(out_degrees, columns=['Nodo', 'Grado de Salida'])
    
    degree = pd.concat([in_degree_df, out_degree_df.drop('Nodo', axis=1)], axis=1)
    degree["Degree"] = degree["Grado de Entrada"] + degree["Grado de Salida"]
    
    return degree

def calculate_centrality(G):
    return nx.degree_centrality(G)

def model(populations, t, growth_rates, predation_rates_dict, predator_mortalities):
    prey_population = populations[0]
    predator_populations = np.array(populations[1:])
    
    d_prey_dt = growth_rates[0] * prey_population
    
    d_predators_dt = []
    for i, predator in enumerate(predator_populations):
        predator_population = predator
        prey_population_for_predator = prey_population
        
        # Sumar todas las interacciones de este depredador con sus presas
        total_predation = sum(predation_rates_dict.get(predator, {}).get(prey, 0) * prey_population for prey in predation_rates_dict.get(predator, []) if prey_population > 0)
        
        d_predator_dt = predator_mortalities[i] * predator - total_predation * prey_population_for_predator
        d_predators_dt.append(d_predator_dt)
    
    return [d_prey_dt] + d_predators_dt + [0]*(10-len(d_predators_dt)-1)  # Add zeros to match the length of y0

def calculate_parameters(data_species, data_interactions):
    interaction_data = {}
    
    for _, row in data_species.iterrows():
        species = row['Nombre_Común']
        growth_rate = row['Tasa_Natalidad']
        initial_population = row['Cantidad_Inicial']
        mortality_rate = row['Tasa_Mortalidad']
        interaction_data[species] = {
            'growth_rate': growth_rate,
            'initial_population': initial_population,
            'mortality_rate': mortality_rate,
            'predators': [],
            'is_prey': False,
            'is_predator': False
        }
    
    for _, row in data_interactions.iterrows():
        predator = row['Depredador']
        prey = row['Presas']
        interaction_data[prey]['predators'].append(predator)
        interaction_data[prey]['is_prey'] = True
        interaction_data[predator]['is_predator'] = True
    
    # Create a dictionary to store predation rates
    predation_rates_dict = {}
    for _, row in data_interactions.iterrows():
        predator = row['Depredador']
        prey = row['Presas']
        predation_rate = row['Tasa de predación']
        if predator not in predation_rates_dict:
            predation_rates_dict[predator] = {}
        predation_rates_dict[predator][prey] = predation_rate
    
    return interaction_data, predation_rates_dict

def run_simulation(data_species, data_interactions):
    interaction_data, predation_rates_dict = calculate_parameters(data_species, data_interactions)
    
    time_steps = np.linspace(0, 100, 1000)  # Simulación de 100 unidades de tiempo

    # Almacenar la evolución de las poblaciones
    populations_over_time = {species: [] for species in interaction_data.keys()}

    # Inicializar poblaciones
    current_populations = []
    for species, params in interaction_data.items():
        current_populations.append(params['initial_population'])

    # Simulación
    for t in time_steps:
        population_solution = odeint(
            model,
            current_populations,
            [0, t],
            args=(
                [params['growth_rate'] for params in interaction_data.values()],
                predation_rates_dict,
                [params['mortality_rate'] for params in interaction_data.values()]
            )
        )[1]  

        # Actualizar la población inicial
        for i, species in enumerate(interaction_data):
            interaction_data[species]['initial_population'] = population_solution[i]
            populations_over_time[species].append(population_solution[i])

        # Imprimir resultados cada instante
        print(f"Tiempo: {t:.2f} - Poblaciones: {', '.join(f'{species}: {p:.2f}' for species, p in zip(interaction_data.keys(), population_solution))}")
        

    # Gráfica de resultados
    plt.figure(figsize=(10, 5))
    for species, populations in populations_over_time.items():
        plt.plot(time_steps, populations, label=species)
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Dinámica de poblaciones de presa y depredador')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    st.title("Modelo de Dinámica de Poblaciones según Holling-Tanner")

    
    data_species, data_interactions = load_data('especies.csv', 'predacion.csv')
    G = create_interaction_graph(data_species, data_interactions)
    plot_graph(G)
    # Calcular y mostrar grados de los nodos
    degree = calculate_degrees(G)
    st.write("## Grado de los nodos")
    st.dataframe(degree)
    
    degree_centrality = calculate_centrality(G)
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_color='red', node_size=2000, font_size=12, ax=ax)
    labels = {node: f"{node}\n{centrality:.2f}" for node, centrality in degree_centrality.items()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
    st.pyplot(fig) 
    st.write("Centralidad de grado:")
    st.write(degree_centrality)

    # Ejecutar simulación de dinámica de poblaciones
    run_simulation(data_species, data_interactions)

if __name__ == "__main__":
    main()
