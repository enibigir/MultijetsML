
import os
import numpy as np
import awkward as ak
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import matplotlib.pyplot as plt

NanoAODSchema.warn_missing_crossrefs = False

input_file = 'output/pentajet-Run3_2023-NANOAOD.root'

def plot(var, xlabel, xmin=0, xmax=3000, bin_width=20):

    bins_by_width = np.arange(xmin, xmax + bin_width, bin_width)

    plt.hist(var, bins_by_width, histtype='step', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Events")
    savname=xlabel.replace('$','').replace('\\','').replace('(','').replace(')','')\
                  .replace('[','').replace(']','')\
                  .replace(' ','')
    
    name=f"{savname}.png"
    plt.savefig(f"plots/{name}")
    print(f"saving plots/{name}")

    plt.show()

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
        

def selection_goodjets(ev, jet_eta_cut: float = 2.4, jet_pt_cut: float = 30):
    """Select good jets"""
    good_jets = (
        (abs(ev.GenJet.eta) < jet_eta_cut)
        & (ev.GenJet.pt > jet_pt_cut)
    )
    selected_jets = ev.GenJet[good_jets]
    n_selected_jets = ak.num(selected_jets)
    event_mask = n_selected_jets >= 0
    selected_events = ev[event_mask]
    return selected_events


def selection_10jets(ev):
    """Select events with at least 10 jets"""
    ev_10j = ev[ak.num(ev.GenJet,axis=1) >= 10]
    return ev_10j

def get_leading_and_subleading(jets):
    ix_jet_pt_2_largest = ak.argpartition(jets, axis=1)
    ordered_2_jets = ak.pad_none(
        jets[ix_jet_pt_2_largest],
        target=2,
        clip=True
    )
    leading_jet = ordered_2_jets[:, 0]
    subleading_jet = ordered_2_jets[:, 1]
    return leading_jet,subleading_jet


def doAnalysis():
    print(f'Input file: {input_file}\n')

    events = NanoEventsFactory.from_root(
        input_file,        # path to the ROOT file
        treepath="Events",     # name of the TTree inside the ROOT file
        schemaclass=NanoAODSchema,  # schema to interpret the NanoAOD structure
    ).events()
    # Signal: gluino -> 5 jets


    good_jets    = selection_goodjets(events, 2.4, 10)
    events_10j   = selection_10jets(good_jets)

    ## cutflow:
    cutflow = {}
    cutflow['all events']    = events
    cutflow['good jets']     = good_jets
    cutflow['>=10 jets']     = events_10j
    print_report(cutflow)

    pentajet = ak.combinations(events_10j.GenJet, 5, fields=["j1","j2","j3","j4","j5"])  # pentajet candidates


    pentajet["p4"] = pentajet.j1 + pentajet.j2 + pentajet.j3 + pentajet.j4 + pentajet.j5  # calculate four-momentum of 5-jet system

    mass_highestPt = pentajet["p4"][ak.argmax(pentajet.p4.pt, axis=1, keepdims=True)].mass
    mass = pentajet["p4"].mass


    pentajet_sorted = pentajet["p4"][ak.argsort(pentajet.p4.pt, axis=1, ascending=False)]
    # leading_mass = pentajet_sorted[:, 0].mass
    # subleading_mass = pentajet_sorted[:, 1].mass

    do_plots=True
    if do_plots:
        plot(ak.flatten(mass),"Invariant mass [all 5-jet](GeV)")
        plot(ak.flatten(mass_highestPt),"Invariant mass [highest pT](GeV)")


if __name__ == "__main__":
    doAnalysis()
