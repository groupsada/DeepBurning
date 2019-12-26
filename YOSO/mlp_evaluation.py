from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, FCLayer
import sys
import StringIO
from collections import OrderedDict
import time

from nn_dataflow.core import Cost
from nn_dataflow.core import MapStrategy, MapStrategyEyeriss
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import NNDataflow
from nn_dataflow.core import Option
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource

from nn_dataflow.version import get_version



def stats_dict(dfsch, cost):
    '''
    Get the stats as an OrderedDict from the NNDataflowScheme.
    '''
    stats = OrderedDict()

    ## Basic stats.

    stats['total_cost'] = dfsch.total_cost
    stats['total_time'] = dfsch.total_time

    stats['total_ops'] = dfsch.total_ops
    stats['total_accesses'] = dfsch.total_accesses
    stats['total_noc_hops'] = dfsch.total_noc_hops

    ## Cost breakdown.

    total_op_cost = dfsch.total_ops * cost.mac_op
    total_access_cost = sum(a * c for a, c
                            in zip(dfsch.total_accesses, cost.mem_hier))
    total_noc_cost = dfsch.total_noc_hops * cost.noc_hop
    total_static_cost = dfsch.total_node_time * cost.unit_static

    sum_cost = total_op_cost + total_access_cost + total_noc_cost \
            + total_static_cost
    assert abs(sum_cost / dfsch.total_cost - 1) < 0.001

    stats['total_op_cost'] = total_op_cost
    stats['total_access_cost'] = total_access_cost
    stats['total_noc_cost'] = total_noc_cost
    stats['total_static_cost'] = total_static_cost

    ## Other stats.

    stats['active_node_pes'] = dfsch.perlayer_stats('active_node_pes')
    stats['total_dram_bandwidth'] = dfsch.perlayer_stats('total_dram_bandwidth')
    stats['schedules'] = dfsch.res_dict

    return stats

def MLP_network(input_size,hiden_fc1,hiden_fc2,hiden_fc3,output_size):
    NN = Network('MLP_L')

    NN.set_input(InputLayer(input_size,1))
    NN.add('fc1',FCLayer(input_size,hiden_fc1))
    NN.add('fc2',FCLayer(hiden_fc1,hiden_fc2))
    NN.add('fc3',FCLayer(hiden_fc2,hiden_fc3))
    NN.add('fc4',FCLayer(hiden_fc3,output_size))

    return NN

class TestMLP_network():
    def __init__(self,mlp_network):
        self.net = mlp_network #MLP_network(18,32,64,32,2)
        self.map_strategy = MapStrategyEyeriss
        self.resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                        dim=PhyDim2(1, 1),
                                                        type=NodeRegion.PROC),
                                 data_regions=(NodeRegion(origin=PhyDim2(0, 0),
                                                          dim=PhyDim2(1, 1),
                                                          type=NodeRegion.DATA),
                                              ),
                                 dim_array=PhyDim2(16, 16),
                                 size_gbuf=128 * 1024 // 2,  # 128 kB
                                 size_regf=512 // 2,  # 512 B
                                )

        self.cost = Cost(mac_op=1,
                         mem_hier=(200, 6, 2, 1),
                         noc_hop=0,
                         unit_static=0)

        self.options = Option()

    def test_eyeriss_isca16(self):
        network = self.net 
        batch_size = 16
        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)

        tops, cache_stats = nnd.schedule_search(self.options)

        if not tops:
            sys.stderr.write("No valid dataflow found!")
            return None
        dfsch = tops[0]

        ## Write results.

        res_map = OrderedDict()

        res_map['net'] = "MLP_L"
        res_map['batch'] = batch_size
        res_map['resource'] = self.resource._asdict()
        res_map['cost'] = self.cost._asdict()
        res_map['options'] = self.options._asdict()

        res_map['cache_stats'] = cache_stats


        stats = stats_dict(dfsch, self.cost)
        for key, val in stats.items():
            res_map[key] = val

        return res_map


def print_res(res_map):
    print("total_cost: %d"%res_map['total_cost'])
    print("total_time: %d"%res_map['total_time'])
    print("total_ops: %d"%res_map['total_ops'])
    print("total_accesses: ",res_map['total_accesses'])
    print("total_noc_hops: %d"%res_map['total_noc_hops'])
    print("total_op_cost: %d"%res_map['total_op_cost'])
    print("total_access_cost: %d"%res_map['total_access_cost'])
    print("total_noc_cost: %d"%res_map['total_noc_cost'])

if __name__ == '__main__':
    start = time.time()
    NN = MLP_network(18,36,36,36,2)
    test_mlp_data_flow = TestMLP_network(NN)
    res_map = test_mlp_data_flow.test_eyeriss_isca16()
    last = time.time()
    print_res(res_map)
    print(float(last-start))
