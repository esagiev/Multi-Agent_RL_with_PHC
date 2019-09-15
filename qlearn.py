#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Developed by Erkin Sagiev, esagiev.github.io
# Version on 15 September 2019.

import numpy as np # To avoid "np is not defined" error.

def PHC(game, reward, gamma, alpha_param, delta_param, iter_max,
        explore_rate, WoLF):
    """
    Win or Learn Fast Policy Hill-Climbing algrorithm
    is taken from Bowling, M. and Veloso, M., 2002.
    Multiagent learning using a variable learning rate.
    Artificial Intelligence, 136(2), pp.215-250.
    """
    
    plen = game.agent_set().size
    alen = game.action_set().size
    pset = game.agent_set()
    apam = alpha_param
    tmax = iter_max
    eps = explore_rate
    tol = 10**(-5) # Tolerance in comparison.
    
    if WoLF is True:
        wpam = delta_param[:2]
        lpam = delta_param[-1]
    else:
        dfix = delta_param
    
    def alfa(t, apam):
        return 1/(apam[0]+apam[1]*t)
    
    def del_W(t, wpam):
        return 1/(wpam[0]+wpam[1]*t)
    
    def del_L(t, wpam, lpam):
        return lpam*1/(wpam[0]+wpam[1]*t)
    
    def norm(pdf):
        return pdf/sum(pdf.T, 0).reshape(plen, 1)
    
    def val_fun(game, acts, vals, reward, gamma, apam, t):
        val_max = game.payoff()[pset, game.best_resp(acts), np.flip(acts)]
        x = np.ones([plen, alen]) # To update only for actions played.
        x[pset, acts] = 1-alfa(t, apam)
        vals = x*vals
        vals[pset, acts] += alfa(t, apam)*(reward + gamma*val_max)
        return vals
    
    def avr_fun(game, acts, hist, t):
        avr_pdf = np.mean(hist[:, :, :(t+1)], 2)
        avr_pdf = avr_pdf + 1/(t+1)*(hist[:,:,t]-avr_pdf)
        return norm(avr_pdf)
    
    def del_C(agent, vals, pdf, avr_pdf, wpam, lpam, t):
        eval_cur = pdf[agent, :] @ vals[agent, :]
        eval_avr = avr_pdf[agent, :] @ vals[agent, :]
        if (eval_cur - eval_avr > -tol):
            return del_W(t, wpam)
        else:
            return del_L(t, wpam, lpam)
    
    def step(game, agent, del_C, vals, acts, pdf):
        delta = np.minimum(pdf[agent, :], del_C/(alen-1)*np.ones(alen))
        act_good = np.argmax(vals[agent])
        if int(acts[agent]) is int(act_good):
            return sum(np.delete(delta, acts[agent]))
        else:
            return -delta[acts[agent]]
    
    
    hist = np.zeros([plen, alen, tmax])
    acts = np.zeros((plen), dtype=int)
    vals = np.zeros([plen, alen])
    dvec = np.ones([plen])
    
    pdf = 1/alen*np.ones([plen, alen])
    pdf_start = pdf.copy()
    
    for t in range(tmax):
        # Getting next policy with exploration.
        x = np.random.choice(2, 1, p=[eps, 1-eps])
        pdf_gen = x*pdf.copy() + (1-x)*1/alen*np.ones([plen, alen])
        for x in range(plen):
            acts[x] = np.random.choice(alen, 1, p=list(pdf_gen[x]))
        
        # Getting Q values.
        vals = val_fun(game, acts, vals, reward, gamma, apam, t)
        
        # Getting learning rates.
        hist[pset, :, t] = pdf[pset, :]
        if WoLF is True:
            pdf_avr = avr_fun(game, acts, hist, t)
            for x in range(plen):
                dvec[x] = del_C(x, vals, pdf, pdf_avr, wpam, lpam, t)
        else:
            dvec = dfix*dvec
        
        # Updating of policies.
        for x in range(plen):
            pdf[x, acts[x]] += step(game, x, dvec[x], vals, acts, pdf)
        pdf = norm(pdf)
    
    return (np.round(pdf_start, 4), np.round(pdf, 4), np.round(hist, 4))

