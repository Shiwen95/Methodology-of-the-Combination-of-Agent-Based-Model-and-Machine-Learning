# Methodology-of-the-Combination-of-Agent-Based-Model-and-Machine-Learning
The global pandemic crisis has prompted the writing of this study. In the United Kingdom, the COVID-19 outbreak has lasted for nearly two years. Many scholars have spent a great deal of time studying the COVID-19 in order to produce a good prediction on future data. Rather than using real-world data from complex scenarios, I created a standard epidemiological agent-based model and used the data set that resulted from it. Despite the fact that the data set is a simplified form of real data, it is capable of representing COVID-19 infection characteristics. The paper’s main purpose is to determine if it’s possible to combine an agent-based model with a machine learning technique. Logistic regression, decision tree regression, random forest regression, linear regression, neural network, k-nearest neighbour, and support vector machine are among the machine learning techniques used.

Agent-based simulation can model real-world scenarios by designing behaviours of interacting agents in the dynamic network. A typical structure of the agent-based model is represented in figure below. In an agent-based model, an agent is associated with its attributes and its methods. Agent attributes can be static or dynamic, such as an agent’s state. Agent methods can be behaviours under certain conditions or fixed patterns of behaviours.

<img src='https://github.com/Shiwen95/Methodology-of-the-Combination-of-Agent-Based-Model-and-Machine-Learning/blob/main/Schematic%20Figure%20of%20ABM.png'>

Agent-based modelling can be implemented by mesa package in Python: https://github.com/projectmesa/mesa/wiki.

The design of the agent-based model is mainly about the infection process in which the virus spreads by infecting agents’ neighbours. The schematic figure below shows how agents in the constructed epidemiological model have been designed. Each agent refers to a person as a unit of infection. The agents have 7 attributes. The agent’s methods include trying to infect neighbours, isolating infected agents, trying to cure infected agents that are detected and trying to vaccinate healthy agents.

<img src="https://github.com/Shiwen95/Methodology-of-the-Combination-of-Agent-Based-Model-and-Machine-Learning/blob/main/ABM.png">


