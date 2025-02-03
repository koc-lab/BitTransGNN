from bitbertgcn_ops_calc import bert_ops, gcn_ops
from typing import Optional

def tot_energy(addop, multop, dtype, processor):
    if dtype == "int32":
        if processor == "7nm":
            addeng = 0.03e-12
            multeng = 1.48e-12
        elif processor == "45nm":
            addeng = 0.1e-12
            multeng = 3.1e-12
    elif dtype == "int8":
        if processor == "7nm":
            addeng = 0.007e-12
            multeng = 0.07e-12
        elif processor == "45nm":
            addeng = 0.03e-12
            multeng = 0.2e-12
    elif dtype == "float32":
        if processor == "7nm":
            addeng = 0.38e-12
            multeng = 1.31e-12
        elif processor == "45nm":
            addeng = 0.9e-12
            multeng = 3.7e-12
    return addeng*addop, multeng*multop

def bert_energy_info(model_type, model_size, exact=True, inv=False):
    highbit = 32.0
    bit_list = [32.0, 1.0, 2.0]
    processor_list = ["7nm", "45nm"]
    for processor in processor_list:
        for bits in bit_list:
            if bits == 32.0:
                dtype_list = ["float32"]
            else:
                dtype_list = ["int8", "int32", "float32"]
            
            addop_high, multop_high = bert_ops(model_type, model_size, highbit)
            add_energy_high, mult_energy_high = tot_energy(addop_high, multop_high, "float32", processor)
            total_energy_high = add_energy_high + mult_energy_high
            for dtype in dtype_list:
                print("----------")
                print("bits: ", bits)
                print("dtype: ", dtype)
                print("processor: ", processor)
                addop, multop = bert_ops(model_type, model_size, bits)
                add_energy, mult_energy = tot_energy(addop, multop, dtype, processor)
                total_energy = add_energy + mult_energy
                if bits == highbit:
                    print("add energy: ", add_energy)
                    print("mult energy: ", mult_energy)
                    print("total energy: ", total_energy)
                else:             
                    if exact:
                        print("add energy: ", add_energy)
                        print("mult energy: ", mult_energy)
                        print("total energy: ", total_energy)
                    else:
                        if inv:
                            print("add energy: ", add_energy_high/add_energy)
                            print("mult energy: ", mult_energy_high/mult_energy)
                            print("total energy: ", total_energy_high/total_energy)
                        else:
                            print("add energy: ", add_energy/add_energy_high)
                            print("mult energy: ", mult_energy/mult_energy_high)
                            print("total energy: ", total_energy/total_energy_high)

def get_bert_total_energy(model_type, model_size, bits, dtype, processor = "7nm", exact=True, inv=False, add_gcn=False, gcn_conf: Optional[dict] = None):
    if add_gcn:
        assert (gcn_conf is not None)
    addop, multop = bert_ops(model_type, model_size, bits)
    add_energy, mult_energy = tot_energy(addop, multop, dtype, processor)
    total_energy = add_energy + mult_energy
    if add_gcn:
        num_nodes, num_edges, dh, dg, do, gcn_bits = gcn_conf["num_nodes"], gcn_conf["num_edges"], gcn_conf["dh"], gcn_conf["dg"], gcn_conf["do"], gcn_conf["gcn_bits"]
        gcn_add_ops, gcn_mult_ops = gcn_ops(num_nodes, num_edges, dh, dg, do, bits=gcn_bits)
        gcn_add_energy, gcn_mult_energy = tot_energy(gcn_add_ops, gcn_mult_ops, dtype, processor)
        gcn_energy = gcn_add_energy + gcn_mult_energy
        total_energy += gcn_energy
    return total_energy