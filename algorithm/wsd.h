
#ifndef SAMPLING_WSD_H
#define SAMPLING_WSD_H


#include "def.h"


typedef struct edge_attr_t {
    double weight{0}, rand{0}, rank{0};

    edge_attr_t() = default;

    edge_attr_t(double weight, double rand, double rank) {
        this->weight = weight;
        this->rand = rand;
        this->rank = rank;
    }

} edge_attr_t;


typedef struct reservoir_atom_t {
    edge_t edge;
    double rank;

    reservoir_atom_t(const edge_t& edge, double rank) {
        this->edge = edge;
        this->rank = rank;
    }

    bool operator<(const reservoir_atom_t& reservoir_atom) const {
        return this->rank < reservoir_atom.rank;
    }

} reservoir_atom_t;



class reservoir_t : public set<reservoir_atom_t> {

public:
    double tau_e{0.}, tau_s{0.};

    bool add(const reservoir_atom_t& reservoir_atom);
};


res_t wsd_triangle_estimate(const vector<stream_atom_t>& stream);
res_t wsd_wedge_estimate(const vector<stream_atom_t>& stream);


#endif //SAMPLING_WSD_H
