using Pkg
using ITensors
Pkg.activate("CT")
using CT
using Random
# using BenchmarkTools



# N = 5  # number of sites
# d = 2   # local degree of freedom for each site
# idx = siteinds(d,N)
# T = emptyITensor(idx)
# T[1,1,1,1,1]=1
# M=MPS(T,idx;cutoff=1e-10,maxdim=20)

# U2=ITensor(reshape(collect(1:16.),(2,2,2,2)),idx[2],idx[4],idx[2]',idx[4]')

# CT.apply_op!(M,U2)

function run_example(L,p,seed)
    # ct=CT.CT_MPS(L=L,seed=seed,x0=0//32,folded=true,store_op=true,store_vec=true)
    # ct=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=true,store_vec=true,ancilla=0)
    ct=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=0,xj=Set([0]),debug=true)
    i=1
    for idx in 1:2*ct.L^2
    # for idx in 1:div(ct.L^2,2)
    # for idx in 1:1
        # println("Run:$idx")
        i=CT.random_control!(ct,i,p,)
    end
    return ct
    # O=CT.Z(ct)
    # EE=CT.von_Neumann_entropy(ct.mps,ct.L÷2)
    # EE=CT.von_Neumann_entropy(ct.mps,1)
    # return O,EE
end


ct=run_example(4, 0,  3)


@show Array(CT.mps_to_tensor(ct.mps),ct.qubit_site)

# ct=CT.CT_MPS(L=4,seed=3,folded=false,store_op=true,store_vec=true,ancilla=0)
# CT.S!(ct,4,MersenneTwister(3);builtin=true)
# CT.S!(ct,4,MersenneTwister(3);builtin=true)
# @show Array(CT.mps_to_tensor(ct.mps),ct.qubit_site)

# @profview run_example(4, 0,  3)

# @btime run_example(20, 1,  10)



# mps=randomMPS(siteinds("Qubit",4),)
# CT.advance_link_tags!(mps,1)

# ct=CT.CT_MPS(L=4,seed=2,x0=0//32,folded=false)
# CT.S!(ct,4;rng=ct.rng_C)