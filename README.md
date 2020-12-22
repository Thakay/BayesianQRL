# Bayesian RL for 'NChain-v0' Enviornment
An Implementation of Bayesian RL in OpenAI Gym to reproduce the results of the paper by **Dearden et al. (1998)** 

To see the code with descreption please visit the the Jupyter Notebook Folder.
For Raw Code Visit the Code folder.

**To Run the Code** simply Execute the *Run.py* at your terminal with python: python3 Run.py

To modify the number of *episodes or the steps* taken change the input method of run() at the end: *run(Num_ep, Num_steps)*

## Notes:
- **Consider due to the Gym Env limitaion you can not set number of steps taken at each run more than 1000! to change that you need some advanced settings, which you can find online.**
- **NChain-v0 enviornment does NOT have a done situation! unless it meets the 1000 steps. if you use this for other envs note to use a break properly.**
- **Nchain-v0 has no render method.** 
- **Code has some bugs which the learning is not always consistent! I am not an expert! so do not assume this is as a reliable source!**



