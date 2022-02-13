
#include "def.h"
#include <cassert>

int rand_int(int l, int u) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<> dist(l, u);
    return dist(gen);
}

double rand_real(double l, double u) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<> dist(l, u);
    return dist(gen);
}

long rand_geo(double p) {
    random_device rd;
    mt19937_64 gen(rd());
    geometric_distribution<long> dist(p);
    return dist(gen);
}

void GetCurTime(struct rusage* curTime) {
    int ret = getrusage(RUSAGE_SELF, curTime);
    if (ret != 0) {
        fprintf(stderr, "The running time info couldn't be collected successfully.\n");
        exit(0);
    }
}


double absolute_relative_error(const res_t& est, const res_t& act) {
    assert(est.counters.size() == act.counters.size());
    return fabs(est.counters[est.counters.size()-1] - act.counters[act.counters.size()-1]) / act.counters[act.counters.size()-1];
}


double mean_absolute_relative_error(const res_t& est, const res_t& act) {
    assert(est.counters.size() == act.counters.size());
    double sum = 0;
    for (unsigned i = 0; i < est.counters.size(); ++i) {
        sum += fabs(est.counters[i] - act.counters[i]) / (act.counters[i] + 1);
    }
    return (double) sum / est.counters.size();
}


res_t expectation( const vector<res_t>& results) {
    res_t final;
    for (unsigned i = 0; i < results[0].counters.size(); ++i) {
        double sum = 0;
        for (const auto & result : results) {
            sum += result.counters[i];
        }
        final.counters.push_back((double) sum / NUM_TEST);
    }

    double runtime = 0;
    for (const auto & result : results) {
        runtime += result.runtime;
    }
    final.runtime = runtime / NUM_TEST;

    return final;
}


// Unit: ms
double GetTime(struct rusage* timeStart, struct rusage* timeEnd /*, double* userTime, double* sysTime */) {
    double userTime = ((float)(timeEnd->ru_utime.tv_sec - timeStart->ru_utime.tv_sec)) * 1e3 +
        ((float)(timeEnd->ru_utime.tv_usec - timeStart->ru_utime.tv_usec)) * 1e-3;
    /* double sysTime = ((float)(timeEnd->ru_stime.tv_sec - timeStart->ru_stime.tv_sec)) * 1e3 +
       ((float)(timeEnd->ru_stime.tv_usec - timeStart->ru_stime.tv_usec)) * 1e-3; */
    return userTime;
}

