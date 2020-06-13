import pandas as pd

from networkx import nx
from random import randrange, random

def sim(n, deg, leng, pcontact, pdeath):
    p = deg/n
    g = nx.gnp_random_graph(n=n+1,p = p)
    suspectible = set(list(range(1,n+1)))
    agent0 = randrange(1,n+1)
    suspectible.remove(agent0)
    infected = {agent0:1}
    state = pd.DataFrame(data={'s':[n-1],'i':[1],'r':[0],'d':[0]})


    to_infect = []

    tick=1 
    while len(infected) > 0:
        to_infect=[]
        cur_dead = 0
        cur_recovered = 0
        print(pcontact)
        for a in suspectible:
            nei = g.neighbors(a)
            for b in g.neighbors(a):
                if random() < pcontact and b in infected.keys():
                    to_infect.append(a)

        for b in list(infected.keys()):
            nei = g.neighbors(b)
            for a in g.neighbors(b):
                if random() < pcontact and a in infected.keys():
                    to_infect.append(a)
            if infected[b] + leng > tick:
                if random() > pdeath:
                    cur_dead +=1
                else:
                    cur_recovered +=1
                del infected[b]
        print('cur_dead: ',cur_dead)
        tick +=1
        for a in to_infect:
            if a in suspectible:
                suspectible.remove(a)
                infected[a] = tick
        
        r = state['r'].iloc[-1] + cur_recovered
        d = state['d'].iloc[-1] + cur_dead
        df2 = pd.DataFrame(data={'s':[len(suspectible)],
                'i':[len(infected)],'r':[r],'d':[d]})

        state = state.append(df2)

        
    return state


if __name__ == "__main__":
    df = pd.DataFrame(data={'pcontact':[],'death':[]})
    for i in range(0,105,5):
        for j in range(1):  
            pcontact = i/100
            #print('sim: ',sim(10000, 10, 5, pcontact, 0.04)['d'].iloc[-1] / 10000)
            df = df.append(pd.DataFrame(data={'pcontact':[pcontact], 'death':[(sim(10000, 10, 5, pcontact, 0.04)['d'].iloc[-1] / 10000)]}))


    print(df)
