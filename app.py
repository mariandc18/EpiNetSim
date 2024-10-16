import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objects as go
import random
import collections
from models.SIR_model import spreading_init, spreading_seed, spreading_make_sir_model, spreading_step
from models.SIS_model import spreading_make_sis_model
from models.SIRS_model_in_period import spreading_make_sirs_model
from models.SIRS_model_psusceptible import spreading_make_sirs_model
from models.SIRD_model import spreading_make_sird_model
from models.SEIRS_model_pLossImmunity import spreading_make_seirs_model
from models.SEIR_model import spreading_make_seir_model
from models.SEIRS_model_immunity_period import spreading_make_seirs_model


SPREADING_SUSCEPTIBLE = 'S'
SPREADING_INFECTED = 'I'
SPREADING_RECOVERED = 'R'
SPREADING_DEAD = 'D'
SPREADING_QUARANTINED = 'Quarantined'
SPREADING_EXPOSED = 'E'

color_map = {
    SPREADING_SUSCEPTIBLE: 'blue',
    SPREADING_INFECTED: 'red',
    SPREADING_RECOVERED: 'green',
    SPREADING_DEAD: 'black',
    SPREADING_QUARANTINED: 'orange',
    SPREADING_EXPOSED:'yellow'
}
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Simulador de Propagación de Epidemias"),
    html.Div([
        html.Div([
            html.Label("Selecciona un Modelo:"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Susceptible - Infectado - Recuperado (SIR)', 'value': 'SIR'},
                    {'label':'Susceptible - Infectado - Recuperado - Fallecido (SIRD)', 'value':'SIRD'},
                    {'label': 'Susceptible - Infectado - Susceptible (SIS)', 'value': 'SIS'},
                    {'label': 'Susceptible - Infectado - Recuperado - Susceptible (SIRS)[Con tiempo de recuperación fijo]', 'value':'SIRS_period'},
                    {'label': 'Susceptible - Infectado - Recuperado - Susceptible (SIRS)[Con probabilidad de susceptibilidad]','value':'SIRS_probability'},
                    {'label':'Susceptible - Expuesto - Infectado - Recuperado (SEIR)', 'value':'SEIR'},
                    {'label': 'Susceptible - Expuesto - Infectado - Recuperado - Susceptible (SEIRS)[Con tiempo de recuperación fijo]', 'value':'SEIRS_inmunity'},
                    {'label': 'Susceptible - Expuesto - Infectado - Recuperado - Susceptible (SEIRS)[Con probabilidad de susceptibilidad]', 'value':'SEIRS_plossimmunity'},
                ],
                value='SIR'
            ),
            html.Div(id='model-parameters'),
        ], id='left-column'),
        html.Div([
            html.Div(id='report', style={'whiteSpace': 'pre-wrap'}),
            html.Button('Iniciar', id='start-button', n_clicks=0),
            html.Button('Pausar', id='pause-button', n_clicks=0),
            html.Button('Continuar', id='continue-button', n_clicks=0),
            html.Button('Reset', id='reset-button', n_clicks=0),
            
            dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True),
            dcc.Graph(id='graph', style={'width': '80vw', 'height': '80vh'}),
            dcc.Store(id='graph-data', data={'graph': None, 'step': 0}),
            dcc.Store(id='simulation-running', data=False),

            # Quarantine, Disconnect Nodes, and Vaccination Sections
            html.Label('Número de Nodos a Poner en Cuarentena:'),
            html.Div([
                dcc.Input(id='quarantine-input', type='number', value=10, min=0),
                html.Button('Cuarentena', id='quarantine-button', n_clicks=0),
            ], style={'display': 'flex', 'alignItems': 'center'}),

            html.Label('Número de Nodos a Desconectar:'),
            html.Div([
                dcc.Input(id='disconnect-input', type='number', value=10, min=0),
                html.Button('Desconectar Nodos', id='disconnect-button', n_clicks=0),
            ], style={'display': 'flex', 'alignItems': 'center'}),

            html.Label('Número de Nodos a Vacunar:'),
            dcc.Input(id='vaccination-input', type='number', value=10, min=0),
            html.Label('Probabilidad de Infección de Nodos Vacunados (pVaccinate):'),
            dcc.Input(id='pVaccinate', type='number', value=0.01, step=0.01),
            html.Button('Vacunar', id='vaccinate-button', n_clicks=0),
        ], id='right-column'),
    ], id='app-container')
])




@app.callback(
    Output('model-parameters', 'children'),
    Input('model-dropdown', 'value')
)
def display_model_parameters(selected_model):
    if selected_model == 'SIR':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),

            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),

            html.Label("Probabilidad de Recuperación:"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
        ])
    elif selected_model == 'SIS':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),

            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),

            html.Label("Probabilidad de Ser Nuevamente Susceptible:"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
        ])
    elif selected_model == 'SIRD':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),
            
            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),
            
            html.Label("Probabilidad de Recuperación:"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
            
            html.Label("Probabilidad de Muerte:"),
            dcc.Input(id='pDeath', type='number', value=0.005, step=0.001),
    ])
    elif selected_model == 'SIRS_period':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),
            
            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),
            
            html.Label("Probabilidad de Recuperación:"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
            
            html.Label("Número de iteraciones para la recuperación:"),
            dcc.Input(id='recovery_duration', type='number', value=5, step=1),
    ])
    
    elif selected_model == 'SIRS_probability':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),
            
            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),
            
            html.Label("Probabilidad de Recuperación: "),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
            
            html.Label("Probabilidad de volver a ser susceptible:"),
            dcc.Input(id='pSusceptible', type='number', value=0.05, step=0.01),
    ])
    
    elif selected_model == 'SEIR':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),
            
            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),
            
            html.Label("Probabilidad de Recuperación:"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
            
            html.Label("Probabilidad de convertirse en infeccioso:"),
            dcc.Input(id='pExposedToInfectious', type='number', value=0.05, step=0.01),
    ])
    
    elif selected_model == 'SEIRS_inmunity':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),
            
            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),
            
            html.Label("Probabilidad de Recuperación:"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
            
            html.Label("Probabilidad de convertirse en infeccioso:"),
            dcc.Input(id='pExposedToInfectious', type='number', value=0.05, step=0.01),
            
            html.Label("Periodo de tiempo de inmunidad:"),
            dcc.Input(id='immunity_period', type='number', value=0.05, step=0.01),
    
    ])
    
    elif selected_model == 'SEIRS_plossimmunity':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),
            
            html.Label("Probabilidad de Infección:"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),
            
            html.Label("Probabilidad de Recuperación:"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
            
            html.Label("Probabilidad de convertirse en infeccioso:"),
            dcc.Input(id='pExposedToInfectious', type='number', value=0.05, step=0.01),
            
            html.Label("Probabilidad de volver a ser susceptible:"),
            dcc.Input(id='pSusceptible', type='number', value=0.05, step=0.01),
  
    ])
  

    else:
        return html.Div() 
    
REPORT_FILE = 'simulation_report.txt'
GRAPH_DATA_FILE = 'nodes_evolution.txt'

def clear_report_files():
    with open(REPORT_FILE, 'w') as file:
        file.write('') 
    with open(GRAPH_DATA_FILE, 'w') as file:
        file.write('')  
clear_report_files()

@app.callback(
    Output('report', 'children'),
    Input('interval', 'n_intervals'),
    State('graph-data', 'data'),
    State('model-dropdown', 'value')
)
def update_report(n_intervals, graph_data, selected_model):
    if graph_data['graph'] is None:
        return ''

    g = nx.node_link_graph(graph_data['graph'])
    state_counts = collections.Counter(nx.get_node_attributes(g, 'state').values())
    report = f"Modelo seleccionado: {selected_model}\n"
    report += f"Iteración: {graph_data['step']}\n"
    for state, count in state_counts.items():
        report += f"{state}: {count} nodos\n"

    # Guarda el reporte en el archivo
    with open(REPORT_FILE, 'a') as file:
        file.write(report + '\n')

    return report


def quarantine_simulation(g, num_quarantine):
    nodes = list(g.nodes)
    quarantined_nodes = random.sample(nodes, min(num_quarantine, len(nodes)))  

    for node in quarantined_nodes:
        g.nodes[node]['state'] = SPREADING_QUARANTINED

    return g

def disconnect_random_nodes(g, num_disconnect):
    nodes = list(g.nodes)
    nodes_to_disconnect = random.sample(nodes, min(num_disconnect, len(nodes)))  

    for node in nodes_to_disconnect:
        edges_to_remove = list(g.edges(node))
        g.remove_edges_from(edges_to_remove)

    return g

def vaccinate_nodes(g, num_vaccinate, p_vaccinate):
    eligible_nodes = [node for node in g.nodes if g.nodes[node]['state'] in [SPREADING_SUSCEPTIBLE, SPREADING_INFECTED] and not g.nodes[node].get('vaccinated', False)]
    vaccinated_nodes = random.sample(eligible_nodes, min(num_vaccinate, len(eligible_nodes)))  
    for node in vaccinated_nodes:
        g.nodes[node]['state'] = SPREADING_SUSCEPTIBLE 
        g.nodes[node]['pInfect'] = p_vaccinate  
        g.nodes[node]['vaccinated'] = True  

    return g

@app.callback(
    Output('graph', 'figure'),
    Output('graph-data', 'data'),
    Output('simulation-running', 'data'),
    Output('interval', 'disabled'),  
    Input('interval', 'n_intervals'),
    Input('start-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    Input('continue-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('quarantine-button', 'n_clicks'),
    Input('disconnect-button', 'n_clicks'),
    Input('vaccinate-button', 'n_clicks'),
    Input('model-dropdown', 'value'),
    State('num-nodes', 'value'),
    State('pInfect', 'value'),
    State('pRecover', 'value'),
    State('quarantine-input', 'value'),
    State('disconnect-input', 'value'),
    State('vaccination-input', 'value'),
    State('pVaccinate', 'value'),
    State('graph-data', 'data'),
    State('simulation-running', 'data')
)    

def update_graph(n_intervals, start_clicks, pause_clicks, continue_clicks,
                  reset_clicks, quarantine_clicks,
                  disconnect_clicks, vaccinate_clicks,
                  selected_model, 
                  num_nodes, p_infect, p_recover,
                  num_quarantine, num_disconnect,
                  num_vaccinate, p_vaccinate,
                  graph_data, simulation_running):
    p_death = 0.005
    recovery_duration=5
    pSusceptible=0.05
    pExposedToInfectious=0.05
    immunity_period=5
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'reset-button':
        return go.Figure(), {'graph': None, 'step': 0}, False, True

    if triggered_id == 'start-button' and graph_data['graph'] is None:
        g = nx.erdos_renyi_graph(num_nodes, 0.01)
        spreading_init(g)
        spreading_seed(g, 0.05)
        graph_data['graph'] = nx.node_link_data(g)
        graph_data['step'] = 1
        simulation_running = True
        return dash.no_update, graph_data, True, False  

    if graph_data['graph'] is not None:
        g = nx.node_link_graph(graph_data['graph'])

    if triggered_id == 'pause-button':
        simulation_running = False
        return dash.no_update, graph_data, simulation_running, True 

    if triggered_id == 'continue-button':
        simulation_running = True
        return dash.no_update, graph_data, simulation_running, False 

    if triggered_id == 'quarantine-button':
        g = quarantine_simulation(g, num_quarantine) 
        graph_data['graph'] = nx.node_link_data(g)

    if triggered_id == 'disconnect-button':
        g = disconnect_random_nodes(g, num_disconnect)  
        graph_data['graph'] = nx.node_link_data(g)

    if triggered_id == 'vaccinate-button':
        g = vaccinate_nodes(g, num_vaccinate, p_vaccinate)  
        graph_data['graph'] = nx.node_link_data(g)

    if simulation_running:
        if selected_model == 'SIR':
            model = spreading_make_sir_model(p_infect, p_recover)
        elif selected_model == 'SIS':
            model = spreading_make_sis_model(p_infect, p_recover)
        elif selected_model == 'SIRD':
            model = spreading_make_sird_model(p_infect, p_recover, p_death)
        elif selected_model == 'SIRS_period':
            model = spreading_make_sirs_model(p_infect, p_recover, recovery_duration)
        elif selected_model == 'SIRS_probability':
            model = spreading_make_sirs_model(p_infect, p_recover, pSusceptible)
        elif selected_model == 'SEIR':
            model = spreading_make_seir_model(p_infect, pExposedToInfectious, p_recover)
        elif selected_model == 'SEIRS_inmunity':
            model = spreading_make_seirs_model(p_infect, pExposedToInfectious,p_recover, immunity_period)
        elif selected_model == 'SEIRS_plossimmunity':
            model = spreading_make_seirs_model(p_infect, pExposedToInfectious,p_recover, pSusceptible)


        spreading_step(g, model)
        graph_data['graph'] = nx.node_link_data(g)
        graph_data['step'] += 1

    pos = nx.spring_layout(g, k=0.6)
    
    node_trace = go.Scatter(
        x=[pos[i][0] for i in g.nodes],
        y=[pos[i][1] for i in g.nodes],
        mode='markers',
        marker=dict(
            color=[color_map[g.nodes[i]['state']] for i in g.nodes],
            size=10,
            line=dict(width=2)
        ),
        name= 'Nodos'
    )
    
    edge_x = []
    edge_y = []
    
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1,color='black'),
        hoverinfo='none',
        mode='lines',
        name= 'Aristas'
    )

    description_map = {
    'S': 'Susceptible',
    'I': 'Infectado',
    'R': 'Recuperado',
    'D': 'SPREADING_DEAD',
    'Quarantined': 'En cuarentena',
    'E': 'Expuesto'
}

    legend_entries = [
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color, size=10),
            name=description_map[state]  
        )
        for state, color in color_map.items()
    ]

    fig = go.Figure(data=[edge_trace,node_trace] + legend_entries )
    
    fig.update_layout(
        showlegend=True,
        xaxis=dict(showgrid=False , zeroline=False),
        yaxis=dict(showgrid=False , zeroline=False)
    )

    return fig , graph_data , simulation_running , False  

if __name__ == '__main__':
   app.run_server(debug=True)