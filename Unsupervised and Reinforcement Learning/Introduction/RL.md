 # Reinforcement Learning
 
### A.What is Reinforcement Learning?

   RL is an area of machine learning that deals with sequential decision-making, aimed at reaching a desired goal. An RL problem is constituted by a decision-maker called an Agent and the physical or virtual world in which the agent interacts, is known as the Environment. The agent interacts with the environment in the form of Action which result in an effect. As a result, the environment will feedback to the agent a new State and Reward. These two signals are the consequences of the action taken by the agent. In articular, the reward is a value indicating how good or bad the action was, and the state is the current representation of the agent and the environment. 

___

### B. Reinforcement Learning vs Supervised Learning



___
### C. Prerequisites


   #### 1.Neural Networks -

   * Neural networks are the agents that learn to map state-action pairs to rewards. They do so by finding and using the right coefficients/weights to approximate the function
connecting inputs to outputs by iteratively adjusting those coefficients/weights along the gradients that promise less or no error. Deep Learning algorithms help develop artificial neural networks which imitate the neuron networks in the human brain. For example, it could be used to distinguish between females and males in images by classifying and clustering the image data such as distances between the shapes and other specifics in the existing photos to predict the identification in a new set of images.

   #### 2.Python –
 
   * It is a general-purpose, interactive, high-level, object-oriented scripting/programming language designed to be highly readable. It is the major code language for Machine Learning & Artificial Intelligence for reasons that it has a low barrier, a great library ecosystem within itself, platform independence, provides flexibility and has a range of good visualization options to choose from. Hence, there is no doubt that Python is the most popular choice among data scientists. 

   #### 3.Probability –
    
   * Whenever data is utilized in a system without a sole logic, the level of uncertainty grows and probability becomes even more relevant as it is the science of quantifying uncertain things. Common sense is introduced into the Deep Learning system by applying probability theorems. The concepts of Conditional Probability, Bell-Curve Model of Normal Distribution and Bayes’ Theorem are more prominent in Reinforcement Learning as compared to other probability models used.
    
   #### 4.Dynamic Programming – 
    
   It is a method used for solving complex problems by breaking them down into sub-problems, finding solutions for these sub-problems and then combining them again to solve the overall problem. The two prerequisites for using Dynamic Programming are:
   
   * Overlapping subproblems: The solutions of the sub-problems that can be cached and reused as they recur many times in the entire process.
   
   * Optimal substructure: The optimal solutions of the sub-problems that can be used to solve the overall problem.It is worthwhile to mention the use of mathematical frameworks here like Markov Decision Processes (MDPs) that solve Reinforcement Learning problems by satisfying the properties of the Bellman Equation and the Value function. Read the beginner’s guide to Deep Reinforcement Learning to clarify concepts here.
       
   #### 5.Linear Algebra – 
   
   * It is a branch of mathematics that lets you describe the coordinates and the interactions of planes in higher dimensions concisely, and thereby perform operations on them. It is useful in Machine Learning in the sense that one can describe complex operations using the formalisms, notations and matrix factorization from linear algebra.Using Linear Algebra in Reinforcement Learning can serve as building blocks for deeper intuition. You can even get more out of the existing algorithms, implement algorithms from scratch, devise new algorithms, the possibilities are endless. 

___

### D. Types of Reinforcement Learning

   There are mainly two types of reinforcement learning, which are:
     
   ####  Positive Reinforcement:
        
   The positive reinforcement learning means adding something to increase the tendency that expected behavior would occur again. It impacts positively on the behavior of the agent and increases the strength of the behavior.This type of reinforcement can sustain the changes for a long time, but too much positive reinforcement may lead to an overload of states that can reduce the consequences.
        
   ####  Negative Reinforcement:
       
   The negative reinforcement learning is opposite to the positive reinforcement as it increases the tendency that the specific behavior will occur again by avoiding the negative condition.It can be more effective than the positive reinforcement depending on situation and behavior, but it provides reinforcement only to meet minimum behavior.
        
 ___       

### E. Elements of Reinforcement Learning

   There are four main elements of Reinforcement Learning, which are given below
        
   1.	Policy
   2.	Reward Signal
   3.	Value Function
   4.	Model of the environment
        
   #### 1) Policy: *A policy can be defined as a way how an agent behaves at a given time. It maps the perceived states of the environment to the actions taken on those states. A policy is the core element of the RL as it alone can define the behavior of the agent. In some cases, it may be a simple function or a lookup table, whereas, for other cases, it may involve general computation as a search process. It could be deterministic or a stochastic policy:*
   *For deterministic policy: a = π(s)*\
   *For stochastic policy: π(a | s) = P[At =a | St = s]*\
                    
   #### 2) Reward Signal: *The goal of reinforcement learning is defined by the reward signal. At each state, the environment sends an immediate signal to the learning agent, and this signal is known as a reward signal. These rewards are given according to the good and bad actions taken by the agent. The agent's main objective is to maximize the total number of rewards for good actions. The reward signal can change the policy, such as if an action selected by the agent leads to low reward, then the policy may change to select other actions in the future.*
 
 #### 3) Value Function: *The value function gives information about how good the situation and action are and how much reward an agent can expect. A reward indicates the immediate signal for each good and bad action, whereas a value function specifies the good state and action for the future. The value function depends on the reward as, without reward, there could be no value. The goal of estimating values is to achieve more rewards.*
    
 #### 4) Model: *The last element of reinforcement learning is the model, which mimics the behavior of the environment. With the help of the model, one can make inferences about how the environment will behave. Such as, if a state and an action are given, then a model can predict the next state and reward.The model is used for planning, which means it provides a way to take a course of action by considering all future situations before actually experiencing those situations. The approaches for solving the RL problems with the help of the model are termed as the model-based approach. Comparatively, an approach without using a model is called a model-free approach*


___
### F. Reinforcement Learning in Action



___

### G. Applications of Reinforcement Learning

   #### 1.	Robotics:
        
   RL is used in Robot navigation, Robo-soccer, walking, juggling, etc.
                
   #### 2.	Control:
        
   RL can be used for adaptive control such as Factory processes, admission control in telecommunication, and Helicopter pilot is an example of reinforcement learning.
       
   #### 3.	Game Playing:
        
   RL can be used in Game playing such as tic-tac-toe, chess, etc.
        
   #### 4.	Chemistry:
             
   RL can be used for optimizing the chemical reactions.
       
   #### 5.	Business:
            
   RL is now used for business strategy planning.
       
   #### 6.	Manufacturing:
            
   In various automobile manufacturing companies, the robots use deep reinforcement learning to pick goods and put them in some containers.
   
   #### 7.	Finance Sector:
           
   The RL is currently used in the finance sector for evaluating trading strategies.

___

### H. Resources to Learn Further

   https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs
        
   http://rail.eecs.berkeley.edu/deeprlcourse/ 
        
   https://www.freecodecamp.org/news/an-introduction-to-reinforcement-learning-4339519de419/ 
       
   https://katefvision.github.io/ 
       
   https://www.cse.iitm.ac.in/~ravi/courses/Reinforcement%20Learning.html 
    
   http://web.stanford.edu/class/cs234/index.html 
   
   
   
   @ Bindu
