import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from numpy.linalg import inv
from scipy.integrate import odeint

class SatEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
  def __init__(self):
    self.mass = 5
    self.Q0 = np.random.rand(4).transpose()
    self.w0 = np.random.rand(3).transpose()
    self.I =  self.getMOIMat(N=3)
    self.T = np.zeros(3).transpose()
    self.ticks = 0.
    self.dt = 0.1
    self.w = np.random.rand(3).transpose()
    self.q = np.random.rand(4).transpose()
    self.done = False
    self.thresh = 1e-6
    self.action_space = 7
    self.observation_space = np.concatenate((self.w,self.q))
     

  def getMOIMat(self,N=3):
      I = np.random.random_integers(-2,2,size=(N,N))
      I = (I + I.T)/2 
      for i in range(N):
            if I[i][i] < 0:
              I[i][i] *= -1
              temp = I[i][i]
              I[i] /=temp
              I[i][i] = temp
            elif I[i][i] == 0:
              I[i][i] = np.argmax(I[i]) + 1
      return np.matrix(I)

  def convertAction(self,action):
    actionHash = np.asarray([[0.,0.,0.],[-1e-2,0.,0.],[1e-2,0.,0.],[0.,-1e-2,0.],[0.,1e-2,0.],[0.,0.,-1e-2],[0.,0.,1e-2]])
    return actionHash[action]

  def getError(self):
    wi = (self.w[0]**2 + self.w[1]**2 + self.w[2]**2)**0.5
    w0 = (self.w0[0]**2 + self.w0[1]**2 + self.w0[2]**2)**0.5
    return abs(wi-w0)


  def getReward(self):
    wi = (self.w[0]**2 + self.w[1]**2 + self.w[2]**2)**0.5
    w0 = (self.w0[0]**2 + self.w0[1]**2 + self.w0[2]**2)**0.5
    return (1/(2*np.pi)**0.5)*np.exp(-0.5* (wi-w0)**2)

  def reset(self):
    self.T = np.zeros(3).transpose()
    self.ticks = 0.
    self.dt = 0.01
    self.w = np.random.rand(3).transpose()
    self.q = np.random.rand(4).transpose()
    self.done = False
    return np.concatenate((self.w,self.q))


  def step(self,action):
    self.ticks += self.dt
    self.T = self.convertAction(action)
    #W updated in self.w
    self.getW()
    #Q updated in self.q
    self.getQ()

    error = self.getError()
    reward = self.getReward()
    if error < self.thresh:
      self.done = True
    else:
      pass

    obs = np.concatenate((self.w,self.q))

    return [obs,reward,self.done,{}]


  def getQ(self,):
    t = np.linspace(0,self.ticks,int(self.ticks/self.dt))
    sol = np.asarray(odeint(self.qvec, self.Q0, t))
    self.q = sol[-1].transpose()
    

  def qvec(self,q,t):
    q0,q1,q2,q3 = self.Q0
    temp_w = 0.5*np.asarray([0,self.w[0],self.w[1],self.w[2]])
    q = np.matrix([[q0, q1, -q2 , -q3],[q1, q0, -q3, q2],[q2, q3, q0, -q1],[q3, -q2, q1, q0]])
    return np.asarray(np.dot(q,temp_w)).reshape(4)



  def getW(self,):
    t = np.linspace(0,self.ticks,int(self.ticks/self.dt))
    sol = np.asarray(odeint(self.wvec, self.w0, t))
    self.w = sol[-1].transpose()
    
  def wvec(self,w, t):
    wx,wy,wz = w
    w_temp = np.array([wx,wy,wz]).transpose()
    return np.asarray(np.dot(inv(np.matrix(self.I)),np.transpose(self.T-(np.cross(w_temp,np.dot(self.I,w_temp)))))).reshape(3)

  
