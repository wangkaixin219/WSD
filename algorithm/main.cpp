#include <iostream>
#include "data.h"
#include "def.h"
#include "wsd.h"



int main(int argc, const char* argv[]) {

    vector<stream_atom_t> stream;

    // real dataset
    if (string(argv[1]) == "real") {
        cout << "Graph file: " << argv[2] << endl << "Deletion type: " << argv[3] << endl;
        stream = generate(argv[2], argv[3]);
    }
    // synthetic dataset
    else {
        cout << "G(" << argv[2] << ", " << argv[3] << ")" << endl << "Deletion type: " << argv[4] << endl;
        stream = generate(stol(argv[2]), stod(argv[3]), argv[4]);
    }

    cout << "Generated (" << stream.size() << ")" << endl << endl;

    res_t actual_triangle = triangle(stream);
    cout << "Triangle " << actual_triangle.counters[actual_triangle.counters.size()-1] << " " << actual_triangle.runtime << "ms" << endl << endl;

    // wsd
    vector<res_t> wsd_triangles;
    for (int i = 0; i < NUM_TEST; ++i) {
        res_t tmp = wsd_triangle_estimate(stream);
        wsd_triangles.push_back(tmp);
        cout << i << " " << (long) tmp.counters[tmp.counters.size()-1] << " " << tmp.runtime << "ms" << endl;
    }
    res_t wsd_triangle_est = expectation(wsd_triangles);
    wsd_triangle_est.absolute_relative_error = absolute_relative_error(wsd_triangle_est, actual_triangle);
    wsd_triangle_est.mean_absolute_relative_error = mean_absolute_relative_error(wsd_triangle_est, actual_triangle);

    cout << "Avg " << (long) wsd_triangle_est.counters[wsd_triangle_est.counters.size()-1] << endl
         <<  "ARE " << wsd_triangle_est.absolute_relative_error * 100 << "%" << endl
         <<  "MARE " << wsd_triangle_est.mean_absolute_relative_error * 100 << "% " << endl
         << "Runtime " << wsd_triangle_est.runtime << "ms" << endl << endl;


    res_t actual_wedge = wedge(stream);
    cout << "Wedge " << actual_wedge.counters[actual_wedge.counters.size()-1] << " " << actual_wedge.runtime << "ms" << endl << endl;

    // wsd
    vector<res_t> wsd_wedges;
    for (int i = 0; i < NUM_TEST; ++i) {
        res_t tmp = wsd_wedge_estimate(stream);
        wsd_wedges.push_back(tmp);
        cout << i << " " << (long) tmp.counters[tmp.counters.size()-1] << " " << tmp.runtime << "ms" << endl;
    }
    res_t wsd_wedge_est = expectation(wsd_wedges);
    wsd_wedge_est.absolute_relative_error = absolute_relative_error(wsd_wedge_est, actual_wedge);
    wsd_wedge_est.mean_absolute_relative_error = mean_absolute_relative_error(wsd_wedge_est, actual_wedge);

    cout << "Avg " << (long) wsd_wedge_est.counters[wsd_wedge_est.counters.size()-1] << endl
         <<  "ARE " << wsd_wedge_est.absolute_relative_error * 100 << "%" << endl
         <<  "MARE " << wsd_wedge_est.mean_absolute_relative_error * 100 << "% " << endl
         << "Runtime " << wsd_wedge_est.runtime << "ms" << endl << endl;


    return 0;
}
