
#ifndef SAMPLING_DATA_H
#define SAMPLING_DATA_H

#include "def.h"

typedef struct pos_t {
    edge_t edge;
    unsigned pos;

    pos_t(const edge_t& edge, unsigned pos) {
        this->edge = edge;
        this->pos = pos;
    }

    bool operator<(const pos_t& p) const {
        return this->pos < p.pos;
    }
} pos_t;

vector<stream_atom_t> generate(long n, double p, const string& del_type);
vector<stream_atom_t> generate(const string& file_name, const string& del_type);
vector<stream_atom_t> massive_deletion(vector<stream_atom_t>& stream, double alpha = ALPHA, double beta_m = BETA_M);
vector<stream_atom_t> light_deletion(vector<stream_atom_t>& stream, double beta_l = BETA_L);
res_t triangle(const vector<stream_atom_t>& stream);
res_t wedge(const vector<stream_atom_t>& stream);


#endif //SAMPLING_DATA_H
