
#include "data.h"
#include "def.h"
#include <algorithm>

edge_t split(const string& line) {
    stringstream ss(line);
    int u, v;
    ss >> u;
    if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
    ss >> v;
    return edge_t(u, v);
}

vector<stream_atom_t> generate(const string& file_name, const string& del_type) {
    ifstream f(file_name);
    string line;
    vector<stream_atom_t> stream;
    map<int, set<int>> V;

    while (getline(f, line)) {
        if (line[0] == '%') continue;
        edge_t edge = split(line);

        if (V[edge.u].count(edge.v) == 0 && V[edge.v].count(edge.u) == 0) {       // ignore duplicates
            stream.emplace_back(op_t::insertion, edge);
            V[edge.u].insert(edge.v);
            V[edge.v].insert(edge.u);
        }
    }
    f.close();

    return  del_type == "massive" ? massive_deletion(stream) : light_deletion(stream);
}


vector<stream_atom_t> generate(long n, double p, const string& del_type) {
    vector<stream_atom_t> stream;
    map<long, vector<long>> V;

    for (unsigned i = 2; i < n; ++i) {
        fprintf(stdout, "Generate %.2lf%% edges\r", (double) i / n * 100);
        fflush(stdout);
        long ambassador = rand_int(1, i-1);
        queue<long> vertex_q;
        set<long> seen;
        vertex_q.push(ambassador);
        seen.insert(i);
        seen.insert(ambassador);

        while (!vertex_q.empty()) {
            long u = vertex_q.front();
            unsigned n_link = rand_geo(p);
            if (V[u].size() <= n_link) {
                for (auto v : V[u]) {
                    if (seen.count(v) == 0) {
                        vertex_q.push(v);
                        seen.insert(v);
                    }

                }
            }
            else {
                while (n_link--) {
                    long v = V[u][rand_int(0, V[u].size()-1)];
                    if (seen.count(v) == 0) {
                        vertex_q.push(v);
                        seen.insert(v);
                    }
                }
            }
            edge_t edge(i, u);
            V[i].push_back(u);
            V[u].push_back(i);   
            stream.emplace_back(op_t::insertion, edge);
            vertex_q.pop();
        }
        seen.clear();
    }
    cout << endl << "|E| = " << stream.size() << endl;
    return del_type == "massive" ? massive_deletion(stream) : light_deletion(stream);
}



vector<stream_atom_t> massive_deletion(vector<stream_atom_t>& src, double alpha, double beta_m) {

    vector<stream_atom_t> dst;
    for (unsigned i = 0, j = 0; i < src.size(); ++i) {
        fprintf(stdout, "Add %.2lf%% deletions\r", (double) i / src.size() * 100);
        fflush(stdout);
        dst.push_back(src[i]);
        if (rand_real(0, 1) < alpha) {        // deletion happens
            for (unsigned k = j; k < i; ++k) {
                if (rand_real(0, 1) < beta_m) {
                    dst.emplace_back(op_t::deletion, src[k].edge);
                }
            }
            j = i;
        }
    }
    cout << endl;
    return dst;
}


vector<stream_atom_t> light_deletion(vector<stream_atom_t>& src, double beta_l) {

    vector<stream_atom_t> dst;
    set<pos_t> edge_pos;

    for (unsigned i = 0; i < src.size(); ++i) {
        fprintf(stdout, "Add %.2lf%% deletions\r", (double) i / src.size() * 100);
        fflush(stdout);
        dst.push_back(src[i]);

        while (!edge_pos.empty() && edge_pos.begin()->pos == i) {
            dst.emplace_back(op_t::deletion, edge_pos.begin()->edge);
            edge_pos.erase(edge_pos.begin());
        }
        if (rand_real(0, 1) < beta_l) {
            unsigned pos = rand_int(i + 1, src.size());
            edge_pos.insert(pos_t(src[i].edge, pos));
        }
    }
    while (!edge_pos.empty()) {
        dst.emplace_back(op_t::deletion, edge_pos.begin()->edge);
        edge_pos.erase(edge_pos.begin());
    }
    cout << endl;
    return dst;
}


res_t triangle(const vector<stream_atom_t>& stream) {

    res_t result;
    double counter = 0;
    map<int, set<int>> V;
    struct rusage start, end;

    GetCurTime(&start);
    for (const auto& atom : stream) {

        int u = atom.edge.u, v = atom.edge.v;

        set<int> common_neighbors;
        set_intersection(V[u].begin(), V[u].end(), V[v].begin(), V[v].end(),
                inserter(common_neighbors, common_neighbors.begin()));

        if (atom.op == op_t::insertion) {
            counter += common_neighbors.size();
            result.counters.push_back(counter);
            V[u].insert(v);
            V[v].insert(u);
        }
        else {
            counter -= common_neighbors.size();
            result.counters.push_back(counter);
            V[u].erase(v);
            V[v].erase(u);
        }
    }
    GetCurTime(&end);
    result.runtime = GetTime(&start, &end);

    return result;
}

res_t wedge(const vector<stream_atom_t>& stream) {
    res_t result;
    double counter = 0;
    map<int, set<int>> V;
    struct rusage start, end;

    GetCurTime(&start);
    for (const auto& atom : stream) {

        int u = atom.edge.u, v = atom.edge.v;

        if (atom.op == op_t::insertion) {
            counter += V[u].size() + V[v].size();
            result.counters.push_back(counter);
            V[u].insert(v);
            V[v].insert(u);
        }
        else {
            counter -= V[u].size() + V[v].size();
            result.counters.push_back(counter);
            V[u].erase(v);
            V[v].erase(u);
        }
    }
    GetCurTime(&end);
    result.runtime = GetTime(&start, &end);

    return result;
}
