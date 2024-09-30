import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objects as go
from models.SIR_model import spreading_init, spreading_seed, spreading_make_sir_model, spreading_step
from models.SIS_model import spreading_make_sis_model


# Constants for SIR model states
SPREADING_SUSCEPTIBLE = 'S'
SPREADING_INFECTED = 'I'
SPREADING_RECOVERED = 'R'
SPREADING_DEAD = 'D'

color_map = {
    SPREADING_SUSCEPTIBLE: 'blue',
    SPREADING_INFECTED: 'red',
    SPREADING_RECOVERED: 'green',
    SPREADING_DEAD: 'black'
}

# Crear la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)


app.layout = html.Div([
    html.H1("Simulador de Propagación de Epidemias"),

    html.Label("Selecciona un Modelo:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'SIR', 'value': 'SIR'},
            {'label': 'SIS', 'value': 'SIS'},
        ],
        value='SIR'
    ),

    html.Div(id='model-parameters'),

    html.Button('Iniciar', id='start-button', n_clicks=0),
    html.Button('Pausar', id='pause-button', n_clicks=0),
    html.Button('Continuar', id='continue-button', n_clicks=0),
    html.Button('Reset', id='reset-button', n_clicks=0),

  
    dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True), 

    dcc.Graph(id='graph', style={'width': '80vw', 'height': '80vh'}),

    dcc.Store(id='graph-data', data={'graph': None, 'step': 0}),
    dcc.Store(id='simulation-running', data=False), 
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

            html.Label("Probabilidad de Infección (pInfect):"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),

            html.Label("Probabilidad de Recuperación (pRecover):"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
        ])
    elif selected_model == 'SIS':
        return html.Div([
            html.Label("Número de Nodos:"),
            dcc.Input(id='num-nodes', type='number', value=300, min=10, step=10),

            html.Label("Probabilidad de Infección (pInfect):"),
            dcc.Input(id='pInfect', type='number', value=0.1, step=0.01),

            html.Label("Probabilidad de Ser Nuevamente Susceptible (pRecover):"),
            dcc.Input(id='pRecover', type='number', value=0.01, step=0.01),
        ])

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
    Input('model-dropdown', 'value'),
    State('num-nodes', 'value'),
    State('pInfect', 'value'),
    State('pRecover', 'value'),
    State('graph-data', 'data'),
    State('simulation-running', 'data')
)
def update_graph(n_intervals, start_clicks, pause_clicks, continue_clicks, reset_clicks, selected_model, num_nodes, p_infect, p_recover, graph_data, simulation_running):
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

    if simulation_running:
        if selected_model == 'SIR':
            model = spreading_make_sir_model(p_infect, p_recover)
        elif selected_model == 'SIS':
            model = spreading_make_sis_model(p_infect, p_recover)

        spreading_step(g, model)
        graph_data['graph'] = nx.node_link_data(g)
        graph_data['step'] += 1

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
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

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
        yaxis=dict(showgrid=False, zeroline=False)
    )

    return fig, graph_data, simulation_running, False  

if __name__ == '__main__':
    app.run_server(debug=True)
