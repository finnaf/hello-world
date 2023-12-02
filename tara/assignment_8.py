#!/usr/bin/env python
# coding: utf-8

# # <center>L2 Computational Physics</center>
# ---
# 
# Please consider consenting to this study surrounding the impact of generative AI on assessent: https://forms.office.com/e/MgshrBLTij
# 
# ## Random Walks
# 
# This assignment will look at some properties of random walks.

# In[1]:

import numpy
import numpy as np
from matplotlib import pyplot as plt 


# To do our work we will implement a walker class. When initialised a list of possible steps is populated. In one dimension it is
# 
# [+s] , [-s] 
# 
# where s is the step size, it defaults to 1 but can be set as an argument in the constructor. In two dimensions the steps list contains
# 
# [ +s , 0 ] , [ -s , 0 ] ,  [ 0 , +s ] , [ 0 , -s ]
# 
# At each step the current position of the walker, saved in `self.pos`, is updated by adding one of the possible steps. The function `pickStep` chooses randomly one of the possible steps. Use this function to implement the `doSteps` function that performs `n` steps and returns a `(n+1) x ndim` array representing the trajectory of the walker, including the starting point. 

# In[34]:


class walker:
    def __init__(self,x0,ndim=1, step_size=1.0):
        self.pos=x0
        self.ndim=ndim
        self.possibleSteps=[]
        for i in range(ndim):
            step=numpy.zeros(ndim)
            step[i]= - step_size
            self.possibleSteps.append(numpy.array(step,dtype='f'))
            step[i]= + step_size
            self.possibleSteps.append(step.copy())
        self.npossible=len(self.possibleSteps)

    def pickStep(self):
        istep = numpy.random.choice(range(self.npossible))
        return self.possibleSteps[istep]
        
    def doSteps(self,n):
        positions=numpy.ndarray((n+1,self.ndim),dtype='f')
        positions[0] = 0
        for i in range(n):
            self.pos = positions[i]
            positions[i+1] = self.pos + self.pickStep()
        return positions


# In[35]:


# this test is worth 2 marks
numpy.random.seed(1111)
w = walker(numpy.zeros(1))
pos_test = w.doSteps(10)
reference = [[ 0.], [-1.], [ 0.], [ 1.], [ 2.], [ 1.], [ 0.], [-1.], [-2.], [-3.], [-4.]]
assert len(pos_test)==11
plt.savefig("fig1.png")
# plots to help debugging
plt.plot(range(11),pos_test, label='your trajectory')
plt.plot(range(11),reference,'o', label='reference points')
plt.legend()
plt.xlabel('step number')
assert (pos_test == reference).all()
plt.savefig("fig2.png")


# In[36]:


# this test is worth 2 marks
# numpy.random.seed(1112)
# w = walker(numpy.zeros(2), ndim=2)
# pos_test = w.doSteps(10)
# reference = numpy.array([[ 0.,  0.], [-1.,  0.], [-1., -1.], [-2., -1.], 
#              [-2.,  0.], [-2.,  1.], [-1.,  1.], [-1.,  2.], 
#              [ 0.,  2.], [ 1.,  2.], [ 0.,  2.]])
# assert pos_test.shape == (11,2)
# # plots to help debugging
# plt.plot(pos_test[:,0], pos_test[:,1],'-o', label='your trajectory')
# plt.plot(reference[:,0],reference[:,1],'o', label='reference points')
# plt.legend()
# assert (pos_test == reference).all()


# This is a plot to visualise trajectories of 10 walkers taking 100 steps.

# In[37]:


nsteps = 100
for i in range(10):
    w = walker(numpy.zeros(1))
    ys = w.doSteps(nsteps)
    print(nsteps, ys)
    plt.plot(range(nsteps+1),ys)
plt.savefig("fig3.png")


# **Task 1**
# 
# Make a plot of average position and average squared position of 100 1D walkers using 1000 steps. Your plot needs a legend, title and labels. [5 marks]
# 

# In[39]:

plt.clf()
### TARAS
# plt.ylabel("distance from start")
# nsteps = 1000
# average_pos = np.zeros((1000, 100))
# for i in range(100):
#     w = walker(numpy.zeros(1))
#     ys = w.doSteps(nsteps)
#     plt.plot(range(nsteps+1), ys)
# plt.savefig("fig4.png")

# Number of walkers
num_walkers = 100

# Number of steps
nsteps = 1000

# Arrays to store positions
average_pos = np.zeros((nsteps + 1, num_walkers))
average_pos_squared = np.zeros((nsteps + 1, num_walkers))

# Run simulations for 100 walkers
for i in range(num_walkers):
    w = walker(np.zeros(1))
    positions = w.doSteps(nsteps)

    # Store individual walker positions
    average_pos[:, i] = positions[:, 0]
    average_pos_squared[:, i] = positions[:, 0] ** 2

# Calculate averages
avg_pos = np.mean(average_pos, axis=1)
avg_pos_squared = np.mean(average_pos_squared, axis=1)

# Plotting
plt.plot(range(nsteps + 1), avg_pos, label="Average Position")
# plt.plot(range(nsteps + 1), avg_pos_squared, label="Average Squared Position")
plt.savefig("average.png")

plt.clf()

# **Task 2**
# 
# Make a plot to show that the average squared distance scaling is independent of the dimension in which the walker moves. 
# Use 100 steps and 400 walkers and use $D=1,2,3,4$. The plot should have a title, legend and labels. [5 marks]
# 

# In[ ]:
nsteps = 100
nwalkers = 400

for j in range(4):
    # 
    all_pos_sq = np.zeros((nsteps+1,nwalkers))

    for i in range(400):
        w = walker(numpy.zeros(j+1), ndim=j+1)
        positions = w.doSteps(nsteps)
        all_pos_sq[:,i] = (positions[:,0])**2

    average_pos_sq = np.mean(all_pos_sq, axis = 1) #axis 1 refers to using the rows rather than the columns
    plt.plot(range(nsteps+1),average_pos_sq, label = f"{j+1}D walker")
   
plt.legend()
plt.title('Average squared position for varying dimensions of random walks')
plt.xlabel('Number of steps')
plt.ylabel('Position')
plt.savefig("task2.png")





# ## 2D walkers

# Use 1000 walkers randomly distributed in the unit square (the positions are given in the array `rand_pos`) simulate the diffusion of particles with step size 0.05. Make a plot of the position of the walkers after 10, 100 and 500 steps. The plots should have labels and titles.
# 
# Tip: Use `plt.scatter` and consider using its `alpha` option to make your plot look nicer. [6 marks]
# 

# In[ ]:


ndim = 2
nwalkers = 500
rand_pos = numpy.random.uniform(size=(nwalkers, ndim))
colours = ['red','green', 'blue']

plt.figure(figsize=(18,6))
for i in range(3): 
    plt.subplot(1, 3, i+1)
    plt.title(f"Plot {i+1}")
    plt.xlim((-3, 4))
    plt.ylim((-3, 4))
    plt.scatter(rand_pos[:,0], rand_pos[:,1], color = colours[i])


# In[ ]:




