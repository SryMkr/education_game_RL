该文件是对open-spiel中的state, observation, agent, policy, environment定义进行理解。open-spiel对于强化学习中的定义精确到了极致，必须仔细深入的研究。  
强化学习中有两点个东西特别关键，第一个是agent，第二个是environment。  
envronment一开始就要定义四个成员 observation，reward，discount，step_type，所以才会有初始化state和make_observation两个方法。其功能就是接受一个action，并随之调整observation中的所有该调整的元素。  
observation由 information state，legal action, current player 三个元素构成，且每一个agent都有其对应的列表   
agent接受一个time_step, 并且将legal action中的动作和计算得到的动作概率一一对应。然后使用抽样的方法随机挑选一个动作，并且返回。   
reward 和discount都是人为设定的参数。  
state的API方法是为了得到一个action之后，生成新的state，帮助env构造新的TimeStep.  


(1) agent     
agent: 接收的是环境给的TimeStep，返回的是action。这个过程要分为两步：（a）接收到TimeStep,agent要处理必要的信息，首先训练策略模型，并给出新的{action:probability/value} (b) 根据给出的动作和概率选择一个动作，并返回    
首先查看rl_agent.py文件。在agent的定义中，只有两个方法 __init__(), step().但是明显少了第一步，那么第一步该如何实现呢？这就涉及到了policy和agent是如何结合的问题！！！       

然后查看rl_agent_policy.py文件。发现其实TimeStep是给了policy的action_probabilities方法。在这个方法中，由分为两步        
(1.1) 将environment中给的TimeStep拆解为current_player可以观察到的信息，并重新构造一个TimeStep      
(1.2) 将这个新的TimeStep输入到agent的step方法，计算动作的概率分布，并且还要完成选择动作      

查看uniform_random_agent.py文件，验证上述结论。发现    
(2.1) 在构造action和policy函数的时候，直接可以返回action和动作的概率密度函数    
(2.2) 在step中，直接返回的就是动作      

综上所述可以得出以下结论，agent确实只有只有两个方法 __init__(), step().      
(3.1) 至于自定义构造action和policy函数，可以给出{action:probability/value}，并且选择动作。紧接着在step中直接读取agent给出的动作      
(3.2) 也可以先构造{action:probability/value}，然后再在step中选择动作        








