# -*- coding: utf-8 -*-
import numpy as np
import math

q = math.log(10) / 400
c = 50
start_rating = 1500
Sigma = 350
Sigma_min = 30

class Glicko(object):
    
    """
    A python object to calculate Glicko 1 ratings

        :param start_rating: determines where the ratings start
        :param Sigma: determines where the rating deviation starts
        :param c: determines degree that ratings deviation increases between periods of evaluation
    """
    
    def __init__(self):
        self.start_rating = start_rating
        self.Sigma = Sigma
        self.q = q
        self.c = c
        
        
    def onset_RD(self, RD_old, t):
        """
        Calculate the onset RD at the beginning of a period
        """
        RD = np.sqrt(np.power(RD_old, 2) + np.power(self.c, 2) * t)
        
        return RD
    
    
    def g(self, RD):
        """
        Calculate 'g' for the opponent
        """
        g = 1 / np.sqrt((1 + 3 * np.power(self.q, 2)) / np.power(np.pi, 2)) 
        
        return g
        
        
    def probability(self, g, Ra, Rb, h=0):
        """
        Calculate expected result
        """
        P = 1 / (1 + np.power(10, -g * ((Ra - Rb) / 400)))
        
        return P
    
    
    def RD_new(self, g, E, RD):
        """
        Calculate post-match RD
        """
        d = 1 / (np.power(self.q, 2) * np.power(g, 2) * E * (1- E))
        RD_new = 1 / np.sqrt(1 / np.power(RD, 2) + 1 / d)
        
        return RD_new
    
    
    def glicko_rating(self, Ra, Rb, RDa, RDb, outcome, h=0, start_rating=start_rating, Sigma=Sigma, c=c):
        """
        Calculate post-match rating
        """
        if start_rating != self.start_rating:
            self.start_rating = start_rating
        
        if Sigma != self.Sigma:
            self.Sigma = Sigma
            
        if c != self.c:
            self.c = c
            
        RDa = self.onset_RD(RDa, 1)
        RDb = self.onset_RD(RDb, 1)
        
        if outcome == 1:
            visitor_outcome = 0
        elif outcome == 0:
            visitor_outcome = 1
        else:
            visitor_outcome = 0.5
            
        g_b = self.g(RDb)
        Pa = self.probability(g_b, Ra, Rb, h)
        if RDa > Sigma_min:
            RDa_new = self.RD_new(g_b, Pa, RDa)
        else:
            RDa_new = RDa
        Ra_post = Ra + self.q * np.power(RDa_new, 2) * g_b * (outcome - Pa)
        
        g_a = self.g(RDa)
        Pb = self.probability(g_a, Rb, Ra, h)
        if RDb > Sigma_min:
            RDb_new = self.RD_new(g_b, Pb, RDb)
        else:
            RDb_new = RDb
        RDb_new = self.RD_new(g_a, Pb, RDb)
        Rb_post = Rb + self.q * np.power(RDb_new, 2) * g_a * (visitor_outcome - Pb)
        
        return Pa, Pb, Ra_post, Rb_post, RDa_new, RDb_new
        
    
        
    
    