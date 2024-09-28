import networkx as nx  
import math  
import numpy.random as rnd  
import matplotlib.pyplot as plt  

# Constants for SIS model states  
SPREADING_SUSCEPTIBLE = 'S'  
SPREADING_INFECTED = 'I'

def spreading_init(g):  
    for i in g.nodes():  
        g.nodes[i]['state'] = SPREADING_SUSCEPTIBLE  

def spreading_seed(g, pSick):  
    for i in g.nodes():  
        if rnd.random() <= pSick:  
            g.nodes[i]['state'] = SPREADING_INFECTED  

def spreading_make_sis_model(pInfect, pRecover):  
    def model(g, i):  
        if g.nodes[i]['state'] == SPREADING_INFECTED:  
            for m in g.neighbors(i):  
                if g.nodes[m]['state'] == SPREADING_SUSCEPTIBLE:  
                    if rnd.random() <= pInfect:  
                        g.nodes[m]['state'] = SPREADING_INFECTED  
        if g.nodes[i]['state'] == SPREADING_INFECTED and rnd.random() <= pRecover:  
            g.nodes[i]['state'] = SPREADING_SUSCEPTIBLE  
    return model  

def spreading_step(g, model):  
    for i in list(g.nodes()):  
        model(g, i)  

def spreading_run(g, model, iterations):  
    for _ in range(iterations):  
        spreading_step(g, model)

# Creación del grafo y simulación
n = 500  
er = nx.erdos_renyi_graph(n, 0.01)  
# Inicializar y sembrar el grafo
spreading_init(er)  
spreading_seed(er, 0.05)  
# Crear y correr el modelo SIS
model = spreading_make_sis_model(0.3, 0.5)  

iterations = 200  
color_map = {  
    SPREADING_SUSCEPTIBLE: 'blue',  
    SPREADING_INFECTED: 'red'  
}

for step in range(0, iterations, 20):  
    spreading_run(er, model, 10)  

    # Contar nodos en cada estado
    count_s = sum(1 for i in er.nodes() if er.nodes[i]['state'] == SPREADING_SUSCEPTIBLE)  
    count_i = sum(1 for i in er.nodes() if er.nodes[i]['state'] == SPREADING_INFECTED)  
    
    print(f"Iteration {step}: S={count_s}, I={count_i}")  

    colors = [color_map[er.nodes[i]['state']] for i in er.nodes()]  

    fig = plt.figure(figsize=(10, 8))  
    ax = fig.gca()  
    ax.grid(False)  
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])  

    pos = nx.spring_layout(er, iterations=50, k=2/math.sqrt(n))  

    nx.draw_networkx_edges(er, pos, width=1, alpha=0.4)  
    nx.draw_networkx_nodes(er, pos, node_size=100, alpha=1,  
                           linewidths=0.5, node_color=colors)  
    plt.title(f"Iteration {step}")  
    plt.show()  
