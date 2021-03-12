# **Reinforcement Learning**

1. What is Reinforcement Learning?

RL is an area of machine learning that deals with sequential decision-making, aimed at reaching a desired goal. An RL problem is constituted by a decision-maker called an  **Agent ** and the physical or virtual world in which the agent interacts, is known as the  **Environment**. The agent interacts with the environment in the form of  **Action** _ ** ** _which results in an effect. As a result, the environment will feedback to the agent a new  **State**  and  **Reward**. These two signals are the consequences of the action taken by the agent. In particular, the reward is a value indicating how good or bad the action was, and the state is the current representation of the agent and the environment. This cycle is shown in the following diagram:

![](RackMultipart20210312-4-dtch9j_html_56f47e24cb1cb5f8.png)

In this diagram the agent is represented by PacMan that based on the current state of the environment, choose which action to take. Its behavior will influence the environment, like its position and that of the enemies, that will be returned by the environment in the form of a new state and the reward. This cycle is repeated until the game ends.

The ultimate goal of the agent is to maximize the total reward accumulated during its lifetime. Let&#39;s simplify the notation: if   ![](RackMultipart20210312-4-dtch9j_html_b08e48bb6bc048fa.png) is the action at time  ![](RackMultipart20210312-4-dtch9j_html_c59ed536b7b985fa.png) and  ![](RackMultipart20210312-4-dtch9j_html_661b23f7f9354f10.png) is the reward at time  ![](RackMultipart20210312-4-dtch9j_html_3e987d8a49f713e2.png), then the agent will take actions  ![](RackMultipart20210312-4-dtch9j_html_2f3f25b5d1c8f684.png), to maximize the sum of all rewards  ![](RackMultipart20210312-4-dtch9j_html_41dabc91978038d.png).

To maximize the cumulative reward, the agent has to learn the best behavior in every situation. To do so, the agent has to optimize for a long-term horizon while taking care of every single action. In environments with many discrete or continuous states and actions, learning is difficult because the agent should be accountable for each situation. To make the problem harder, RL can have very sparse and delayed rewards, making the learning process more arduous.

b. Reinforcement Learning vs Supervised Learning

c. prerequisites

d. types of reinforcement learning

e. elements of reinforcement learning

f. reinforcement learning in action

g. applications of reinforcement learning

h. resources to learn further