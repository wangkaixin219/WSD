
#ifndef SAMPLING_DEF_H
#define SAMPLING_DEF_H


#define NUM_TEST 100
#define MAX_RESERVOIR_SIZE 200000
#define ALPHA 3.333e-7
#define BETA_M 0.8
#define BETA_L 0.2


#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <string>
#include <sys/resource.h>

using namespace std;

typedef enum operation {insertion, deletion} op_t;

typedef struct edge_t {
    int u{}, v{};           // u < v

    edge_t() = default;

    edge_t(int u, int v) {
        this->u = min(u, v);
        this->v = max(u, v);
    }

    edge_t(const edge_t &edge) {
        this->u = edge.u;
        this->v = edge.v;
    }

    bool operator<(const edge_t& edge) const {
        return (long) this->u * INT32_MAX + this->v < (long) edge.u * INT32_MAX + edge.v;
    }

    bool operator==(const edge_t& edge) const {
        return this->u == edge.u && this->v == edge.v;
    }

    edge_t& operator=(const edge_t& edge) = default;

} edge_t;


typedef struct stream_atom_t {
    operation op;
    edge_t edge;

    stream_atom_t(op_t op, const edge_t& edge) {
        this->op = op;
        this->edge = edge;
    }

} stream_atom_t;


typedef struct res_t {
    vector<double> counters;
    double runtime, absolute_relative_error, mean_absolute_relative_error;
} res_t;


int rand_int(int l, int u);
double rand_real(double l, double u);
long rand_geo(double p);
void GetCurTime(struct rusage* curTime);
double GetTime(struct rusage* timeStart, struct rusage* timeEnd /*, double* userTime, double* sysTime */);
res_t expectation( const vector<res_t>& results);
double absolute_relative_error(const res_t& est, const res_t& act);
double mean_absolute_relative_error(const res_t& est, const res_t& act);

#endif //SAMPLING_DEF_H
