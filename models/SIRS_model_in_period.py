import networkx as nx  
import math
import numpy.random as rnd

SPREADING_SUSCEPTIBLE = 'S'
SPREADING_INFECTED = 'I'
SPREADING_RECOVERED = 'R'

def spreading_init(g):
    for i in g.nodes():
        g.nodes[i]['state'] = SPREADING_SUSCEPTIBLE
        g.nodes[i]['recovery_time'] = 0  

def spreading_seed(g, pSick):
    for i in g.nodes():
        if rnd.random() <= pSick:
            g.nodes[i]['state'] = SPREADING_INFECTED

def spreading_make_sirs_model(pInfect, pRecover, recovery_duration):
    def model(g, i):
        if g.nodes[i]['state'] == SPREADING_INFECTED:
            for m in g.neighbors(i):
                if g.nodes[m]['state'] == SPREADING_SUSCEPTIBLE:
                    if rnd.random() <= pInfect:
                        g.nodes[m]['state'] = SPREADING_INFECTED
            
            if rnd.random() <= pRecover:
                g.nodes[i]['state'] = SPREADING_RECOVERED
                g.nodes[i]['recovery_time'] = 0  
        
        elif g.nodes[i]['state'] == SPREADING_RECOVERED:
            g.nodes[i]['recovery_time'] += 1  
            if g.nodes[i]['recovery_time'] >= recovery_duration:
                g.nodes[i]['state'] = SPREADING_SUSCEPTIBLE

    return model

def spreading_step(g, model):
    for i in list(g.nodes()):
        model(g, i)

def spreading_run(g, model, iterations):
    for _ in range(iterations):
        spreading_step(g, model)

