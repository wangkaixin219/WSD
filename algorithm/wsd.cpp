
#include "wsd.h"
#include <algorithm>


bool reservoir_t::add(const reservoir_atom_t& reservoir_atom) {
    if (this->size() == MAX_RESERVOIR_SIZE) {
        this->tau_s = this->begin()->rank;

        if (reservoir_atom.rank > this->tau_s) {
            this->erase(this->begin());
            this->tau_e = this->tau_s;
            this->insert(reservoir_atom);
            return true;
        }
        else if (reservoir_atom.rank > this->tau_e) {
            this->tau_e = reservoir_atom.rank;
        }
    }
    else {
        if (reservoir_atom.rank > this->tau_s) {
            this->insert(reservoir_atom);
            return true;
        }
    }
    return false;
}


res_t wsd_triangle_estimate(const vector<stream_atom_t> &stream) {
    res_t result;
    double counter = 0;
    map<edge_t, edge_attr_t> Es;
    reservoir_t R;                      // begin() is the smallest
    map<int, set<int>> Vs;               // vertices
    struct rusage start, end;

    GetCurTime(&start);
    for (const auto& stream_atom : stream) {
        edge_t e = stream_atom.edge;
        int u = min(e.u, e.v), v = max(e.u, e.v);
        set<int> common_neighbors;
        set_intersection(Vs[u].begin(), Vs[u].end(), Vs[v].begin(), Vs[v].end(),
                         inserter(common_neighbors, common_neighbors.begin()));

        for (const auto& w : common_neighbors) {
            edge_t e1(u, w), e2(w, v);
            double prob1 = min(1., Es[e1].weight / R.tau_e), prob2 = min(1., Es[e2].weight / R.tau_e);

            if (stream_atom.op == op_t::insertion) counter += 1. / (prob1 * prob2);
            else counter -= 1. / (prob1 * prob2);
        }

        result.counters.push_back(counter);

        if (stream_atom.op == op_t::insertion) {
            double rand = rand_real(0, 1), weight = 9. * common_neighbors.size() + 1., rank =  weight / rand;
            reservoir_atom_t reservoir_atom(e, rank);
            edge_attr_t edge_attr(weight, rand, rank);

            if (R.size() == MAX_RESERVOIR_SIZE && rank > R.begin()->rank) {
                edge_t del_edge = R.begin()->edge;
                Vs[del_edge.u].erase(del_edge.v);
                Vs[del_edge.v].erase(del_edge.u);
                Es.erase(del_edge);
            }

            if (R.add(reservoir_atom)) {
                Vs[u].insert(v);
                Vs[v].insert(u);
                Es[e] = edge_attr;
            }
        }
        else {
            if (Es.count(e)) {
                R.erase(reservoir_atom_t(e, Es[e].rank));
                Vs[u].erase(v);
                Vs[v].erase(u);
                Es.erase(e);
            }
        }
    }
    GetCurTime(&end);
    result.runtime = GetTime(&start, &end);

    return result;

}



res_t wsd_wedge_estimate(const vector<stream_atom_t> &stream) {
    res_t result;
    double counter = 0;
    map<edge_t, edge_attr_t> Es;
    reservoir_t R;                      // begin() is the smallest
    map<int, set<int>> Vs;               // vertices
    struct rusage start, end;

    GetCurTime(&start);
    for (const auto& stream_atom : stream) {
        edge_t e = stream_atom.edge;
        int u = min(e.u, e.v), v = max(e.u, e.v);

        for (const auto& w : Vs[u]) {
            edge_t e1(u, w);
            double prob = min(1., Es[e1].weight / R.tau_e);
            if (stream_atom.op == op_t::insertion) counter += 1. / prob;
            else counter -= 1. / prob;
        }

        for(const auto& w : Vs[v]) {
            edge_t e1(v, w);
            double prob = min(1., Es[e1].weight / R.tau_e);
            if (stream_atom.op == op_t::insertion) counter += 1. / prob;
            else counter -= 1. / prob;
        }

        result.counters.push_back(counter);

        if (stream_atom.op == op_t::insertion) {
            double rand = rand_real(0, 1), weight = 9. * Vs[u].size() + 9. * Vs[v].size() + 1., rank =  weight / rand;
            reservoir_atom_t reservoir_atom(e, rank);
            edge_attr_t edge_attr(weight, rand, rank);

            if (R.size() == MAX_RESERVOIR_SIZE && rank > R.begin()->rank) {
                edge_t del_edge = R.begin()->edge;
                Vs[del_edge.u].erase(del_edge.v);
                Vs[del_edge.v].erase(del_edge.u);
                Es.erase(del_edge);
            }

            if (R.add(reservoir_atom)) {
                Vs[u].insert(v);
                Vs[v].insert(u);
                Es[e] = edge_attr;
            }
        }
        else {
            if (Es.count(e)) {
                R.erase(reservoir_atom_t(e, Es[e].rank));
                Vs[u].erase(v);
                Vs[v].erase(u);
                Es.erase(e);
            }
        }
    }
    GetCurTime(&end);
    result.runtime = GetTime(&start, &end);

    return result;
}

