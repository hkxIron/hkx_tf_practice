"""
wiki:https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
The observations (normal, cold, dizzy) along with a hidden state (healthy, fever) form a hidden Markov model (HMM), and can be represented as follows in the Python programming language:

obs = ('normal', 'cold', 'dizzy')
states = ('Healthy', 'Fever')
start_p = {'Healthy': 0.6, 'Fever': 0.4}
trans_p = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
emit_p = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }
In this piece of code, start_probability represents the doctor's belief about which state the HMM is in when the patient first visits (all he knows is that the patient tends to be healthy).
The particular probability distribution used here is not the equilibrium one, which is (given the transition probabilities) approximately {'Healthy': 0.57, 'Fever': 0.43}.
The transition_probability represents the change of the health condition in the underlying Markov chain.
 In this example, there is only a 30% chance that tomorrow the patient will have a fever if he is healthy today.
 The emission_probability represents how likely the patient is to feel on each day.
  If he is healthy, there is a 50% chance that he feels normal; if he has a fever, there is a 60% chance that he feels dizzy.

  @:param obs: list of str
  @:param states:list of str
  @:param start_p:{"key1":float }
  @:param trans_p:{"key1":{"key2":float }}
  @:param emit_p:{"key1":{"key2":float }}

"""
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    # t = 0
    for st in states: # "healthy","fever"
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            # 计算最大的状态转移概率
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    for line in dptable(V): print(line)
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print('The steps of states are: ' + ' '.join(opt) + '\nwith highest probability of: %s' % max_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%8d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

class state:
    Healthy="Healthy"
    Fever="Fever"
class symptom:
    normal="normal"
    cold="cold"
    dizzy="dizzy"

def main():
   states = [state.Healthy,state.Fever]
   obs = [symptom.normal,symptom.cold,symptom.dizzy,symptom.dizzy,symptom.cold]

   start_p={state.Healthy:0.6,state.Fever:0.4}

   trans_p={state.Healthy:{},state.Fever:{}}
   trans_p[state.Healthy][state.Healthy]=0.7
   trans_p[state.Healthy][state.Fever]=0.3
   trans_p[state.Fever][state.Healthy]=0.4
   trans_p[state.Fever][state.Fever]=0.6

   emit_p={state.Healthy:{},state.Fever:{}}
   emit_p[state.Healthy][symptom.normal]= 0.5
   emit_p[state.Healthy][symptom.cold]= 0.4
   emit_p[state.Healthy][symptom.dizzy]= 0.1
   emit_p[state.Fever][symptom.normal]= 0.1
   emit_p[state.Fever][symptom.cold]= 0.3
   emit_p[state.Fever][symptom.dizzy]= 0.6

   viterbi(obs,states,start_p,trans_p,emit_p)

if __name__ == "__main__":
    main()