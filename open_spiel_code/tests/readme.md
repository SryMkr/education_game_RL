该文件是对open-spiel中的state, observation, agent, policy, environment定义进行理解。open-spiel对于强化学习中的定义精确到了极致，必须仔细深入的研究。  
强化学习中有两点个东西特别关键，第一个是agent，第二个是environment。  
envronment一开始就要定义四个成员 observation，reward，discount，step_type，所以才会有初始化state和make_observation两个方法。其功能就是接受一个action，并随之调整observation中的所有该调整的元素。  
observation由 information state，legal action, current player 三个元素构成，且每一个agent都有其对应的列表   
agent接受一个time_step, 并且将legal action中的动作和计算得到的动作概率一一对应。然后使用抽样的方法随机挑选一个动作，并且返回。   
reward 和discount都是人为设定的参数。  
state的API方法是为了帮助环境捕捉当前状态的信息，构造一个TimeStep.  






