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

def lotka_volterra_model(populations, t, growth_rates, predation_rates_dict, predator_mortalities):
    prey_population = populations[0]
    predator_populations = np.array(populations[1:])
    
    d_prey_dt = growth_rates[0] * prey_population
    for predator, prey in predation_rates_dict.items():
        for p, rate in prey.items():
            d_prey_dt -= rate * prey_population * predator_populations[list(predation_rates_dict.keys()).index(predator)]
    
    d_predators_dt = []
    for i, predator in enumerate(predation_rates_dict.keys()):
        d_predator_dt = predator_mortalities[i] * predator_populations[i]
        for p, rate in predation_rates_dict[predator].items():
            d_predator_dt += rate * prey_population * predator_populations[i]
        d_predators_dt.append(d_predator_dt)
    
    return [d_prey_dt] + d_predators_dt + [0]*(10-len(d_predators_dt)-1)  # Add zeros to match the length of y0

def disaster_event(data_species, disaster_type):
    if disaster_type == 'incendio':
        data_species['Tasa_Natalidad'] *= 0.3  
        data_species['Tasa_Mortalidad'] *= 2.0  
    elif disaster_type == 'inundación':
        data_species['Tasa_Natalidad'] *= 0.6  
        data_species['Tasa_Mortalidad'] *= 1.8  
    elif disaster_type == 'sequía':
        data_species['Tasa_Natalidad'] *= 0.5  
        data_species['Tasa_Mortalidad'] *= 1.5  
    return data_species

def resource_availability_event(data_species, available_resources):
    for index, species in data_species.iterrows():
        if available_resources == 'escasos':
            data_species.at[index, 'Tasa_Natalidad'] *= 0.7  
            data_species.at[index, 'Tasa_Mortalidad'] *= 1.2  
        elif available_resources == 'abundantes':
            data_species.at[index, 'Tasa_Natalidad'] *= 1.3  
            data_species.at[index, 'Tasa_Mortalidad'] *= 0.8  
    return data_species

def disease_outbreak(data_species, infected_species):
    for species in infected_species:
        data_species.loc[data_species['Nombre_Común'] == species, 'Tasa_Mortalidad'] *= 2.0
        data_species.loc[data_species['Nombre_Común'] == species, 'Tasa_Natalidad'] *= 0.5
    return data_species

def stochastic_model(populations, t, growth_rates, predation_rates_dict, mortality_rates):
    noise_scale = 0.35  # Escala de la variabilidad
    noisy_growth_rates = np.random.normal(growth_rates, noise_scale * np.array(growth_rates))
    noisy_mortality_rates = np.random.normal(mortality_rates, noise_scale * np.array(mortality_rates))

    # Cálculo de la dinámica de la presa
    prey_population = populations[0]
    predator_populations = np.array(populations[1:])
    
    d_prey_dt = noisy_growth_rates[0] * prey_population
    for predator, prey in predation_rates_dict.items():
        for p, rate in prey.items():
            d_prey_dt -= rate * prey_population * predator_populations[list(predation_rates_dict.keys()).index(predator)]
    
    # Cálculo de la dinámica de los depredadores
    d_predators_dt = []
    for i, predator in enumerate(predation_rates_dict.keys()):
        d_predator_dt = noisy_mortality_rates[i] * predator_populations[i]
        for p, rate in predation_rates_dict[predator].items():
            d_predator_dt += rate * prey_population * predator_populations[i]
        d_predators_dt.append(d_predator_dt)
 
    num_species = len(populations)
    num_predators = len(d_predators_dt)
    
    if num_predators < num_species - 1:  # -1 por la presa
        d_predators_dt += [0] * (num_species - 1 - num_predators)

    return [d_prey_dt] + d_predators_dt


    
def calculate_parameters(data_species, data_interactions):
    interaction_data = {}
    
    for _, row in data_species.iterrows():
        species = row['Nombre_Común']
        growth_rate = row['Tasa_Natalidad']
        initial_population = row['Cantidad_Inicial']
        mortality_rate= row['Tasa_Mortalidad']
        
        interaction_data[species] = {
            'growth_rate': growth_rate,
            'initial_population': initial_population,
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

def climatic_event(data_species, current_temperature, current_humidity):
    
    # Convertir la columna a flotante antes del bucle
    data_species['Humedad_Óptima (%)'] = data_species['Humedad_Óptima (%)'].str.split('-').str.get(0).astype(float)

    for index, species in data_species.iterrows():
        if not (species['Temperatura_Mínima (°C)'] <= current_temperature <= species['Temperatura_Máxima (°C)']):
            data_species.at[index, 'Tasa_Natalidad'] *= 0.5  
            data_species.at[index, 'Tasa_Mortalidad'] *= 1.5 
        
        # Ajuste basado en humedad
        optimal_humidity = species['Humedad_Óptima (%)']
        if abs(current_humidity - optimal_humidity) > 10: 
            data_species.at[index, 'Tasa_Natalidad'] *= 0.7
            data_species.at[index, 'Tasa_Mortalidad'] *= 1.2
    
    return data_species


def plot_dynamic_graph(G, populations_over_time, pos, time_step):
    node_sizes = [populations_over_time[species][time_step] * 100 for species in G.nodes()]
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, width=0.5, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', edgecolors='darkgrey')

    labels = {node: f"{node}\nCantidad: {int(populations_over_time[node][time_step])}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
    plt.title('Grafo de Interacciones en un Ecosistema - Paso de tiempo: {}'.format(time_step), fontsize=15)
    plt.axis('off')
    st.pyplot(plt)

def run_simulation(data_species, data_interactions, model_func):
    interaction_data, predation_rates_dict = calculate_parameters(data_species, data_interactions)
    
    time_steps = np.linspace(0, 100, 1000)  
    populations_over_time = {species: [] for species in interaction_data.keys()}

    # Inicializar poblaciones
    current_populations = []
    for species, params in interaction_data.items():
        current_populations.append(params['initial_population'])

    # Simulación
    for t in time_steps:
        if t % 500 == 0:
            current_temperature = np.random.uniform(0, 40)  # 
            current_humidity = np.random.uniform(0, 100)    
            data_species = climatic_event(data_species, current_temperature, current_humidity)
            print(f"Tiempo: {t:.2f} - Cambios climáticos: Temperatura = {current_temperature:.2f}°C, Humedad = {current_humidity:.2f}%")
            
        growth_rates = [row['Tasa_Natalidad'] for _, row in data_species.iterrows()]
        mortality_rates = [row['Tasa_Mortalidad'] for _, row in data_species.iterrows()]
        
        population_solution = odeint(
            model_func,
            current_populations,
            [0, t],
            args=(
                [params['growth_rate'] for params in interaction_data.values()],
                predation_rates_dict,
                [0.1] * len(current_populations) 
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
    st.title("Modelo de Dinámica de Poblaciones en un Ecosistema")

    
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


    run_simulation(data_species, data_interactions, lotka_volterra_model)
    #run_simulation(data_species, data_interactions, stochastic_model)

if __name__ == "__main__":
    main()