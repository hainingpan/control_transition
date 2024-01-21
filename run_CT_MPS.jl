using ITensors
using Random
using LinearAlgebra
using MKL
using Pkg
Pkg.activate("CT")
using CT

using ArgParse
using Serialization

function run(L::Int,p::Float64,seed::Int,ancilla::Int)
    
    ct_f=CT.CT_MPS(L=L,seed=seed,folded=true,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([0]))
    i=1
    T_max = ancilla ==0 ? 2*(ct_f.L^2) : div(ct_f.L^2,2)

    for idx in 1:T_max
        println(idx)
        i=CT.random_control!(ct_f,i,p)
    end
    O=CT.Z(ct_f)
    max_bond= CT.max_bond_dim(ct_f.mps)
    if ancilla ==0 
        EE=CT.von_Neumann_entropy(ct_f.mps,div(ct_f.L,2))
    else
        EE=CT.von_Neumann_entropy(ct_f.mps,1)
    end
    return O, EE, max_bond
end

function parse_my_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--p", "-p"
        arg_type = Float64
        default = 0.0
        help = "measurement rate"
        "--L", "-L"
        arg_type = Int
        default = 8
        help = "system size"
        "--seed", "-s"
        arg_type = Int
        default = 0
        help = "random seed"
        "--ancilla", "-a"
        arg_type = Int
        default = 0
        help = "number of ancilla"
    end
    return parse_args(s)
end

function main()
    println("Uses threads: ",BLAS.get_num_threads())
    println("Uses backends: ",BLAS.get_config())
    args = parse_my_args()
    results = run(args["L"], args["p"], args["seed"],args["ancilla"])

    filename = "MPS_(0,1)_L$(args["L"])_p$(round(args["p"], digits=2))_s$(args["seed"]).jls"
    open(filename, "w") do f
        serialize(f, Dict("O" => results[1], "EE" => results[2], "max_bond" => results[3],"args" => args))
    end
end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__ && main()

# julia --sysimage ~/.julia/sysimages/sys_itensors.so run_CT_MPS.jl --p 1 --L 8 --seed 0 --ancilla 0