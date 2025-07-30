import sys,time
from datetime import timedelta
import awkward as ak
import numpy as np
import torch
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

NanoAODSchema.warn_missing_crossrefs = False

def print_report(data):
    ev_all= data['all events']
    den  = ak.num(ev_all.GenJet, axis=0)
    denj = ak.sum(ak.num(ev_all.GenJet, axis=1))
    for cut, events in data.items():
        if cut=='all events':
            print('{}: events = {} ({} jets), efficiency = 100%'.format(cut, den, denj))
        else:
            num = ak.num(events.GenJet, axis=0)
            numj = ak.sum(ak.num(events.GenJet, axis=1))
            print('{}: events = {} ({} jets), efficiency = {}%'.format(cut,num, numj, 100 * num/den))


def get_cpu_time(start_cpu_time):
    n = time.process_time() - start_cpu_time
    dt = str(timedelta(seconds = n))
    print(f"CPU time: {dt}")

def PPRINT(msg = None):
    print(f"Line No: {sys._getframe().f_back.f_lineno}: {msg if msg is not None else ''}")


def tight_jets(ev, jet_eta_cut: float = 2.4, jet_pt_cut: float = 30):
    jet_cut = ((abs(ev.GenJet.eta) < jet_eta_cut) & (ev.GenJet.pt > jet_pt_cut))

    tj = ev.GenJet[jet_cut]
    
    ev["GenPart","signal"] = 0

    print("\tGenPart.hasFlags...", end=" ");start_cpu_time = float(time.process_time())
    gp = ev.GenPart[(ev.GenPart.hasFlags(["isLastCopy","isPrompt"]))]
    get_cpu_time(start_cpu_time)

    print("\tGluino decays...", end=" ");start_cpu_time = float(time.process_time())
    slg  = gp[gp.pdgId == 1000021]

    ch1 = slg[:,0].distinctChildren
    ch1["signal"] = 1
    nch1 = ak.flatten(ch1.distinctChildren,axis=2)
    nch1["signal"] = 1
    nnch1 = ak.flatten(nch1.distinctChildren,axis=2)
    nnch1["signal"] = 1
    nnnch1 = ak.flatten(nnch1.distinctChildren,axis=2)
    nnnch1["signal"] = 1
    nnnnch1 = ak.flatten(nnnch1.distinctChildren,axis=2)
    nnnnch1["signal"] = 1
    nnnnnch1 = ak.flatten(nnnnch1.distinctChildren,axis=2)
    nnnnnch1["signal"] = 1

    ch2 = slg[:,1].distinctChildren
    ch2["signal"] = 2
    nch2 = ak.flatten(ch2.distinctChildren,axis=2)
    nch2["signal"] = 2
    nnch2 = ak.flatten(nch2.distinctChildren,axis=2)
    nnch2["signal"] = 2
    nnnch2 = ak.flatten(nnch2.distinctChildren,axis=2)
    nnnch2["signal"] = 2
    nnnnch2 = ak.flatten(nnnch2.distinctChildren,axis=2)
    nnnnch2["signal"] = 2
    nnnnnch2 = ak.flatten(nnnnch2.distinctChildren,axis=2)
    nnnnnch2["signal"] = 2
    get_cpu_time(start_cpu_time)

    print("\tConcatenating...", end=" ");start_cpu_time = float(time.process_time())
    chs = ak.concatenate([ch1, nch1, nnch1,nnnch1,nnnnch1,nnnnnch1, ch2, nch2, nnch2, nnnch2,nnnnch2,nnnnnch2], axis=1)
    get_cpu_time(start_cpu_time)
    
    ev["SlimGenPart"] = gp

    print("\tMatching...", end=" ");start_cpu_time = time.process_time()
    tj["MatchedGenPart"] = tj.nearest(chs,threshold=0.4)
    tj["signal"] = tj.MatchedGenPart.signal
    myslice = (tj.signal == 1) | (tj.signal == 2)
    ev["TightJet"] = tj[ak.fill_none(myslice,False)]
    get_cpu_time(start_cpu_time)


def is_signal_trijet(j1,j2,j3,j4,j5):
    # return j1.signal == j2.signal == j3.signal == j4.signal == j5.signal
    return ( (j1.signal == j2.signal) & (j2.signal == j3.signal) & (j3.signal== j4.signal) & (j4.signal== j5.signal) )

####### BEGIN ##########
input_file = 'output/pentajet-Run3_2023-NANOAOD.root' #str(sys.argv[1])
input_file = 'output/rpvmssm_gogo_10j_m1000-Run3_2023-NANOGEN.root'
print('Input file: {}'.format(input_file))
# file_list = [s.strip("\n") for s in open(input_file)]
# file_list = [input_file]
# print(file_list)
#
# events = NanoEventsFactory.from_root(
#     dict.fromkeys(file_list, "Events"),
#     schemaclass=NanoAODSchema,
#     metadata={"dataset": "DYJets"},
# ).events()

events = NanoEventsFactory.from_root(
    input_file,        # path to the ROOT file
    treepath="Events",     # name of the TTree inside the ROOT file
    schemaclass=NanoAODSchema,  # schema to interpret the NanoAOD structure
).events()

# print("Number of events: ", ak.num(events,axis=0).compute())

print("Number of events: {}".format(len(events)))
cutflow = {}
cutflow['all events']    = events
print_report(cutflow)

print("Filtering Tight Jets...");start_cpu_time = time.process_time()
tight_jets(events)
get_cpu_time(start_cpu_time)

# cut_events = events[ak.num(events.Jet,axis=1) >= 10]
cut_events = events[ak.num(events.TightJet,axis=1) >= 10]
# print("Number of events: ", ak.num(cut_events,axis=0).compute())
print("Number of events with >= 10 jets: {}".format(len(cut_events)))

print(ak.num(cut_events.TightJet))
print("Number of events after num jet selection: ", ak.num(cut_events,axis=0))
print("num of events with 10+ tight jets maybe ",ak.num(cut_events.TightJet,axis=0))


selected_jets = cut_events.TightJet[:,0:10]
print("Number of selected jets: {}".format(len(selected_jets)))

print("Making combinatorics...");start_cpu_time = time.process_time()
trijet = ak.combinations(selected_jets, 5, fields=["j1","j2","j3","j4","j5"])
get_cpu_time(start_cpu_time)
# for i in range(3):
#     print(trijet[i][0])
# exit(1)
result = ak.zip(
    {
        "j1": trijet.j1,
        "j2": trijet.j2,
        "j3": trijet.j3,
        "j4": trijet.j4,
        "j5": trijet.j5,
        "p4": trijet.j1 + trijet.j2 + trijet.j3 + trijet.j4 + trijet.j5,
        # "pt": trijet.j1.pt + trijet.j2.pt + trijet.j3.pt + trijet.j4.pt + trijet.j5.pt,
        # "eta": trijet.j1.eta + trijet.j2.eta + trijet.j3.eta + trijet.j4.eta + trijet.j5.eta,
        # "phi": trijet.j1.phi + trijet.j2.phi + trijet.j3.phi + trijet.j4.phi + trijet.j5.phi,
        # "px": trijet.j1.px + trijet.j2.px + trijet.j3.px + trijet.j4.px + trijet.j5.px,
        # "py": trijet.j1.py + trijet.j2.py + trijet.j3.py + trijet.j4.py + trijet.j5.py,
        # "pz": trijet.j1.pz + trijet.j2.pz + trijet.j3.pz + trijet.j4.pz + trijet.j5.pz,
        # "e": trijet.j1.energy + trijet.j2.energy + trijet.j3.energy + trijet.j4.energy + trijet.j5.energy,
        "match": is_signal_trijet(trijet.j1, trijet.j2, trijet.j3, trijet.j4, trijet.j5),
    },
    with_name="Momentum4D",
)

properties = {
    "pt":  result.p4.pt,
    "eta": result.p4.eta,
    "phi": result.p4.phi,
    "e":   result.p4.energy,
}
match = result.match
mass = result.p4.mass

print("Making NN input...");start_cpu_time = time.process_time()
events = []
print(len(mass))
for j in range(len(mass)):
    if j % 500 == 0:
        print(str(j) + "...")
    event = []
    for i in range(252): # change to pairs instead of triplets
        trip = []
        for key in properties:
            trip.append(properties[key][j][i])

        trip.append(match[j][i])
        trip.append(mass[j][i])
        event.append(trip)
    events.append(event)

data = np.array(events)
get_cpu_time(start_cpu_time)

X = np.array([i.flatten() for i in data[:,:,0:-2]])
Y = data[:,:,-2:-1]
M = data[:,:,-1:]

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
M = torch.tensor(M, dtype=torch.float32)

torch.save(X,"x_tensor.pd")
torch.save(Y,"y_tensor.pd")
torch.save(M,"m_tensor.pd")



