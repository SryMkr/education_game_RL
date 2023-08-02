该文件是对open-spiel中的state, observation, agent, policy, environment定义进行理解。open-spiel对于强化学习中的定义精确到了极致，必须仔细深入的研究。  
强化学习中有两点个东西特别关键，第一个是agent，第二个是environment。  
  
observation由 information state，legal action, current player 三个元素构成，且每一个agent都有其对应的列表   
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
(3.3) 可以直接在step中做拆解行为，以及选择动作      
(3.4) 我的问题在于，在哪训练模型了呢？按照rl_agent给的代码来看，应该是在step中就训练模型了，因为有一个参数是is_evaluation，如果设置为false，也需要得到奖励从而训练模型。      
(3.5) 可以看出对于agent来说，就是就是根据给出的TimeStep,用current_player的integer作为索引，读取该agent应该看到的observation，训练模型，并且选择动作。这也是为什么agent必须读取一个TimeStep的原因    


(2) environment      
environment： 具有四个成员 observation， reward， discount，step_type，其功能就是接受一个action，并调整调整各个成员。该过程分为两步。根据马尔可夫链，环境中主要有一个state概率转移矩阵，而且是必须使用动作才能转移
所以说，其实对于environment的构造分为两步。      
(2.1) 构造所有agent的成员      
environment中的observation成员里面其实还有三个成员information_state, legal_action, current_player.所以在整个environment的结构体中一共有6个成员，而且每个agent都有其对应的成员    
(2.2) 接受动作，并给出相应的改变，这个功能主要通过state来实现      
在state中，接受到一个动作以后，current_player会改变，legal_action也会改变，state也会改变，reward也会改变，还需要判断游戏结束没有，以及step的类型。所以在state层，主要就是为了判断下一个TimeStep变成了什么      
(2.3) environment通过访问state的变化，构造一个新的TimeStep发送给agent.    

(3) 对open-spiel的整体理解。      
(3.1) 在嵌入一个游戏的时候，首先要定义一大堆的游戏参数，其实不为别的，就是为了构造一个初始化的TimeStep      
(3.2) 在嵌入新游戏的时候需要设定state，其实就是定义一个state的转换        
(3.3) 在嵌入游戏的时候，之所以要定义observer, 其实是为了构造TimeStep    
(3.4) environment就像一个总工程师，控制着这个游戏空间的所有元素，state只是为了判断，接受一个动作以后，那些元素需要做出改变，并且在observer中集成总结，并给到TimeStep中。agent其实并不是需要定义好多，agent的所有信息都是根据
TimeStep来的，一个agent的类其实只需要如何读取信息，并做相应的决策即可。
(3.5) open-spiel集成的东西太多，不是我一个人就能完全学会的，所以目前只能理解交互的过程。但是得实时保持学习，尤其是人家得整体框架和各个强化学习得概念是如何交互的，是未来学习的重点框架。
(3.6) open-spiel, rl, nlp, game design, study science 构成接下来10年的主要学习方向。其他的10年以后再说。











