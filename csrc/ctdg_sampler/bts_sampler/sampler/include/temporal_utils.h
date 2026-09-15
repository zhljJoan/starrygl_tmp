#pragma once
#include <torch/extension.h>
#include <parallel_hashmap/phmap.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <map>
#include "head.h"
// #include <boost/thread/mutex.hpp>
using namespace std;

#define work_thread 10
th::Tensor sparse_get_index(th::Tensor in,th::Tensor map_key){
    auto key_ptr = map_key.data_ptr<NodeIDType>();
    auto in_ptr = in.data_ptr<NodeIDType>();
    int sz =  map_key.size(0);
    vector<pair<NodeIDType,NodeIDType>> mp(sz);
    vector<NodeIDType> out(in.size(0));
#pragma omp parallel for
    for(int i=0;i<sz;i++){
        mp[i] = make_pair(key_ptr[i],i);
    }
    phmap::parallel_flat_hash_map<NodeIDType,NodeIDType> dict(mp.begin(),mp.end());

#pragma omp parallel for
    for(int i=0;i<in.size(0);i++){
        out[i] = dict.find(in_ptr[i])->second;
    }
    return th::tensor(out);
}
vector<double> get_norm_temporal(th::Tensor row,th::Tensor col,th::Tensor timestamp,int num_nodes){
    vector<double> ret(4);
    HashM<NodeIDType,TimeStampType> dict0;
    HashM<NodeIDType,TimeStampType> dict1;
    auto rowptr = row.data_ptr<NodeIDType>();
    auto colptr = col.data_ptr<NodeIDType>();
    auto time_ptr = timestamp.data_ptr<TimeStampType>();
    vector<TimeStampType> out_timestamp[work_thread];
    vector<TimeStampType> in_timestamp[work_thread];
#pragma omp parallel for num_threads(work_thread)
    for(int i = 0;i<row.size(0);i++){
        int tid = omp_get_thread_num();
        if(dict0.find(rowptr[i])!=dict0.end()){
            out_timestamp[tid].push_back(time_ptr[i]-dict0.find(rowptr[i])->second);
            dict0.find(rowptr[i])->second = time_ptr[i];
        }
        else dict0.insert(make_pair(rowptr[i],time_ptr[i]));
        if(dict1.find(colptr[i])!=dict1.end()){
            in_timestamp[tid].push_back(time_ptr[i]-dict1.find(colptr[i])->second);
            dict1.find(colptr[i])->second = time_ptr[i];
        }
        else dict1.insert(make_pair(colptr[i],time_ptr[i]));
    }
    double srcavg = 0;
    double dstavg = 0;
    double srcvar = 0;
    double dstvar = 0;
    double srccnt = 0;
    double dstcnt = 0;
    for(int i = 0;i<work_thread;i++){
        #pragma omp parallel for num_threads(work_thread)
        for(auto &v: in_timestamp[i]){
            dstavg += v;
            dstcnt++;
        }
        #pragma omp parallel for num_threads(work_thread)
        for(auto &v: out_timestamp[i]){
            srcavg += v;
            srccnt++;
        }
    }
    dstavg /= dstcnt;  
    srcavg /= srccnt;
    for(int i = 0;i<work_thread;i++){
        #pragma omp parallel for num_threads(work_thread)
        for(int j = 0;j<in_timestamp[i].size();j++){
            TimeStampType v=in_timestamp[i][j];
            dstvar += (v-dstavg)*(v-dstavg)/dstavg;
        }
        #pragma omp parallel for num_threads(work_thread)
        for(int j = 0;j<out_timestamp[i].size();j++){
            TimeStampType v=out_timestamp[i][j];
            srcavg += (v-srcavg)*(v-srcavg)/srcavg;
        }
    }
    ret[0]=srcavg;
    ret[1]=srcvar;
    ret[2]=dstavg;
    ret[3]=dstvar;
    return ret;
}

PYBIND11_MODULE(torch_utils, m)
{
    m
    .def("sparse_get_index", 
        &sparse_get_index, 
        py::return_value_policy::reference)
    .def("get_norm_temporal",
        &get_norm_temporal, 
        py::return_value_policy::reference
    );
}