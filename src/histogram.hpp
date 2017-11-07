#ifndef _HISTOGRAM_HPP
#define _HISTOGRAM_HPP

#include <vector>
#include <iostream>

#include <cmath>

template<class T>
class Histogram {

    const size_t MAX_BINS = 300;
public:

    Histogram(T bin_size) : 
        bin_size(bin_size), 
        max(0),
        nb_bins(0),
        _hist(MAX_BINS,0)
    {
    }

    void add(T val) {
        _vals.push_back(val);

        unsigned int idx = val / bin_size;

        if (idx > MAX_BINS) throw "value beyond MAX_BINS X BIN_SIZE";

        if(idx + 1 > nb_bins) nb_bins = idx + 1;

        _hist[idx] += 1;
        if (_hist[idx] > max) max = _hist[idx];
    }

    float avg() {
        T sum = 0;
        for (auto v : _vals) sum += v;

        return sum / _vals.size();
    }

    float stddev() {

        auto av = avg();

        T sum = 0;

        for (auto v : _vals) sum += (v - av) * (v - av);

        return sqrt(sum/(_vals.size() - 1));

    }

    std::vector<unsigned int> get() {return _hist;}

    T bin_size;

    unsigned int nb_bins;
    unsigned int max;

private:
    std::vector<T> _vals;
    std::vector<unsigned int> _hist;
};

#endif
