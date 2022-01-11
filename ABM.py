import math
from enum import Enum
import networkx as nx

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import pandas as pd
import numpy as np


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2
    DEAD = 3
    ISOLATED = 4 #After identifying being infected

#10k population, actually around 600 million population in 2020 in UK
population = 2000

def number_state(model, s):
    n=0
    for a in model.schedule.agents:
        if a.state == s:
            n+=1
    return n

def number_infected(model):
    #deteced infected and undetected infected
    return number_state(model, State.INFECTED) + number_state(model, State.ISOLATED)


def number_susceptible(model):
    return number_state(model, State.SUSCEPTIBLE)


def number_resistant(model):
    return number_state(model, State.RESISTANT)


def number_dead(model):
    return number_state(model, State.DEAD)

def number_isolated(model):
    return number_state(model, State.ISOLATED)


class VirusOnNetwork(Model):
    """A virus model with some number of agents"""

    def __init__(
        self,
        num_nodes=population,
        avg_node_degree=6,
        initial_outbreak_size=2,
        virus_spread_chance=0.1,
        virus_check_frequency=0.2,
        recovery_chance=0.1,
        curing_days=0,
        testing_days=0,
        time_steps=0,
        wasinfected=False,
        wasresistant=False
    ):

        self.num_nodes = num_nodes
        prob = avg_node_degree / self.num_nodes
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.initial_outbreak_size = initial_outbreak_size 
        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.curing_days = curing_days
        self.testing_days = testing_days
        self.time_steps = time_steps
        self.wasinfected = wasinfected
        self.wasresistant = wasresistant

        self.datacollector = DataCollector(
            {
                "Infected": number_infected,
                "Susceptible": number_susceptible,
                "Resistant": number_resistant,
                "Dead": number_dead,
                "Isolated": number_isolated
            }
        )

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            a = VirusAgent(
                i,
                self,
                State.SUSCEPTIBLE,
                self.virus_spread_chance,
                self.virus_check_frequency,
                self.recovery_chance,
                self.curing_days,
                self.testing_days,
                self.time_steps,
                self.wasinfected,
                self.wasresistant
            )
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)

        # Infect some nodes
        infected_nodes = self.random.sample(self.G.nodes(), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.state = State.INFECTED
            a.wasinfected = True

        self.running = True
        self.datacollector.collect(self)

    def resistant_susceptible_ratio(self):
        try:
            return number_state(self, State.RESISTANT) / number_state(
                self, State.SUSCEPTIBLE
            )
        except ZeroDivisionError:
            return math.inf

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()

#vaccination on daily basis
vr = pd.read_csv('/Users/shiwenyu/Desktop/利兹大学/Dissertation/datasets/vacc_rate.csv')

#testing on daily basis
te = pd.read_csv('/Users/shiwenyu/Desktop/利兹大学/Dissertation/datasets/test_rate.csv')

class VirusAgent(Agent):
    def __init__(
        self,
        unique_id,
        model,
        initial_state,
        virus_spread_chance,
        virus_check_frequency,
        recovery_chance,
        curing_days,
        testing_days,
        time_steps,
        wasinfected,
        wasresistant
    ):
        super().__init__(unique_id, model)

        self.state = initial_state

        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.curing_days = curing_days
        self.testing_days = testing_days
        self.time_steps = time_steps
        self.wasinfected = wasinfected
        self.wasresistant = wasresistant

    def try_to_infect_neighbors(self):
        
        #both susceptible and resistant neighbors could be infected
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        healthy_neighbors = [
            agent
            for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
            if agent.state == State.SUSCEPTIBLE or agent.state == State.RESISTANT
        ]
        
        for a in healthy_neighbors:
            chance = self.decide_virus_spread_chance(a)
                
            if self.random.random() < chance:
                a.state = State.INFECTED
                a.wasinfected = True                    
                    
    def decide_virus_spread_chance(self, neighbor):
        
        #check whether neighbor is immune
        resistant_neighbor = neighbor.wasinfected == True or neighbor.wasresistant == True

        if resistant_neighbor == False:
            chance =  self.virus_spread_chance
        else:
            chance =  self.virus_spread_chance*(1-0.84)
        
        #if the infected self-isolate, the spread chance would decrease
        if self.state == State.ISOLATED:
            chance = chance * 0.3

        #if the infected previously get vaccination, the reproduce chance would reduce as well
        if self.wasresistant == True:
            return chance*0.55
        else:
            return chance


    def step(self): #refer to one day
        
        #infect neighbor #distance #not only 8neighbors
        if self.state == State.INFECTED or self.state == State.ISOLATED:
            self.try_to_infect_neighbors()
        
        #only close contact would be tested and have results after 7 days
        if self.state == State.INFECTED:
            if self.testing_days == 7:
                self.state = State.ISOLATED
                self.testing_days = 0
            else:
                self.testing_days += 1
        
        #cure detected infected people (ISOLATED)
        if self.state == State.ISOLATED:
            if self.curing_days == 30:
                if self.random.random() < self.recovery_chance:
                    # Success
                    self.state = State.SUSCEPTIBLE
                else:
                    # Failed and died after a month
                    self.state = State.DEAD
                self.curing_days = 0
                
            else:
                self.curing_days += 1
        
        #SUSCEPTIBLE gets vaccination
        if self.state == State.SUSCEPTIBLE:
            #vaccination_rate = vr['fullvacci_num'][self.time_steps]/1000000/population
            vaccination_rate = 0.02
            if self.random.random() < vaccination_rate: #quite_low
                self.state = State.RESISTANT
                self.wasresistant = True
            
        self.time_steps += 1 #time pass
        


        
        
        