# Sorting_Network_DRL
![python](https://img.shields.io/badge/Language-Python-pink)

Sorting Network Solver Using Deep Reinforcement Learning.


This project employs Deep Reinforcement Learning (DRL) to tackle the sorting network problem. By leveraging a Deep Q-Network (DQN) and a custom-built environment with a reward system, the project aims to identify optimal sorting network configurations.

Environment and Reward System:

Action : The action that i choose was swapping two elements. 
Reward : The most important part in this algorithm is how the reward was built , given one out of 3 situation: 
1. Swapping made an improvement then a positive reward is given.
2. Swapping didn't make any improvement then a penalty is given.
3. After finshing all actions if the list is sorted correctly then a very large reward is given.
4. After finshing all actions if the list is sorted correctly with the minimum numbers of comparators then a very large reward is given.
5.  After finshing all actions if the list is sorted correctly without the minimum numbers of comparators then a very large penalty is given.

   
Highlights:

1.Utilized Deep Reinforcement Learning to approach the sorting network problem, a novel method compared to traditional optimization techniques.
2.Designed and implemented a custom environment to simulate the sorting network scenarios, complete with a reward mechanism to guide the learning process.
3.Employed a DQN neural network to learn and predict optimal configurations based on the environment's feedback.

