import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objects as go
from models.SIR_model import spreading_init, spreading_seed, spreading_make_sir_model, spreading_step

# Constants for SIR model states
SPREADING_SUSCEPTIBLE = 'S'
SPREADING_INFECTED = 'I'
SPREADING_RECOVERED = 'R'

# Colores para los estados del modelo
color_map = {
    SPREADING_SUSCEPTIBLE: 'blue',
    SPREADING_INFECTED: 'red',
    SPREADING_RECOVERED: 'green'
}

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Crear el layout de la aplicación
app.layout = html.Div([
    html.H1("Simulador de Propagación de Epidemias"),
    
    # Selector para el número de nodos
    html.Label("Número de Nodos:"),
    dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),
    
    # Parámetro de probabilidad de infección
    html.Label("Probabilidad de Infección (pInfect):"),
    dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),
    
    # Parámetro de probabilidad de recuperación
    html.Label("Probabilidad de Recuperación (pRecover):"),
    dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
    
    # Botones de control de la simulación
    html.Button('Iniciar', id='start-button', n_clicks=0),
    html.Button('Pausar', id='pause-button', n_clicks=0),
    
    # Intervalo para las iteraciones automáticas
    dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True),  # Intervalo de 1 segundo
    
    # Gráfico de la simulación
    dcc.Graph(id='graph'),
    
    # Almacenar el estado del grafo y la simulación
    dcc.Store(id='graph-data', data={'graph': None, 'step': 0})
])

# Callback para controlar la simulación y actualizar el gráfico automáticamente
@app.callback(
    Output('graph', 'figure'),
    Output('graph-data', 'data'),
    Output('interval', 'disabled'),
    Input('start-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    Input('interval', 'n_intervals'),
    State('num-nodes', 'value'),
    State('pInfect', 'value'),
    State('pRecover', 'value'),
    State('graph-data', 'data'),
    State('interval', 'disabled')
)
def update_graph(start_clicks, pause_clicks, n_intervals, num_nodes, p_infect, p_recover, graph_data, interval_disabled):
    # Inicializar el grafo si el botón de iniciar fue presionado y no hay grafo en graph_data
    if start_clicks > 0 and graph_data['graph'] is None:
        # Crear un grafo de Erdos-Renyi
        g = nx.erdos_renyi_graph(num_nodes, 0.01)
        spreading_init(g)  # Inicializar los estados de los nodos
        spreading_seed(g, 0.05)  # Infectar una parte de la población
        graph_data['graph'] = nx.node_link_data(g)  # Almacenar el grafo en formato JSON
        graph_data['step'] = 1
        interval_disabled = False  # Iniciar el intervalo automáticamente
    elif graph_data['graph'] is not None:
        # Recuperar el grafo almacenado
        g = nx.node_link_graph(graph_data['graph'])
    else:
        # Si no se ha inicializado nada y no se ha hecho click en "Iniciar"
        return dash.no_update, graph_data, interval_disabled

    # Crear el modelo basado en las probabilidades de infección y recuperación
    model = spreading_make_sir_model(p_infect, p_recover)

    # Continuar la simulación si el intervalo está activo
    if not interval_disabled:
        spreading_step(g, model)  # Ejecutar un paso de la simulación
        graph_data['graph'] = nx.node_link_data(g)  # Actualizar el grafo almacenado
        graph_data['step'] += 1

    # Pausar la simulación si se presiona el botón de pausa
    if pause_clicks > 0:
        interval_disabled = True

    # Crear la visualización del grafo utilizando Plotly
    pos = nx.spring_layout(g)
    node_trace = go.Scatter(
        x=[pos[i][0] for i in g.nodes],
        y=[pos[i][1] for i in g.nodes],
        mode='markers',
        marker=dict(
            color=[color_map[g.nodes[i]['state']] for i in g.nodes],
            size=10,
            line=dict(width=2)
        )
    )
    edge_x = []
    edge_y = []

    # Rellenar las coordenadas de las aristas
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]  
        edge_y += [y0, y1, None]  

    # Ahora creamos el Scatter plot usando las listas de coordenadas
    edge_trace = go.Scatter(
        x=edge_x,  
        y=edge_y,  
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        width=1200,  
        height=600   
    )

    return fig, graph_data, interval_disabled

# Ejecutar la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)
