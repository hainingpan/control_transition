using ITensors
using Random
using LinearAlgebra
using MKL
using Pkg
Pkg.activate("CT")
using CT

using ArgParse
using Serialization
# using Threads

function run(L::Int, p::Float64, seed::Int, ancilla::Int)

    ct_f = CT.CT_MPS(L=L, seed=seed, folded=true, store_op=false, store_vec=false, ancilla=ancilla, debug=false, xj=Set([0]))
    i = 1
    T_max = ancilla == 0 ? 2 * (ct_f.L^2) : div(ct_f.L^2, 2)

    for idx in 1:T_max
        # println(idx)
        i = CT.random_control!(ct_f, i, p)
    end
    O = CT.Z(ct_f)
    max_bond = CT.max_bond_dim(ct_f.mps)
    if ancilla == 0
        EE = CT.von_Neumann_entropy(ct_f.mps, div(ct_f.L, 2))
    else
        EE = CT.von_Neumann_entropy(ct_f.mps, 1)
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
    # BLAS.set_num_threads(14)
    println("Uses threads: ", BLAS.get_num_threads())
    println("Uses backends: ", BLAS.get_config())
    # args = parse_my_args()

    # L_list = 8:2:10
    L_list = 8:2:16
    # p_list = 0.0:0.5:1.0
    p_list = 0.0:0.05:1.0
    seed_list = 1:2000
    ancilla_list = [0, 1]

    O_map = zeros(length(L_list), length(p_list), length(seed_list), length(ancilla_list))
    EE_map = zeros(length(L_list), length(p_list), length(seed_list), length(ancilla_list))
    max_bond_map = zeros(length(L_list), length(p_list), length(seed_list), length(ancilla_list))

    for (L_idx, L) in enumerate(L_list)
        for (p_idx, p) in enumerate(p_list)
            for (ancilla_idx, ancilla) in enumerate(ancilla_list)
                println("L=$L, p=$p, ancilla=$ancilla")
                Threads.@threads for seed in seed_list
                    results = run(L, p, seed, ancilla)
                    O_map[L_idx, p_idx, seed, ancilla_idx] = results[1]
                    EE_map[L_idx, p_idx, seed, ancilla_idx] = results[2]
                    max_bond_map[L_idx, p_idx, seed, ancilla_idx] = results[3]
                end
            end
        end
    end

    filename = "MPS_(0,1)_L($(L_list[1])-$(L_list[end]))_p($(p_list[1])-$(p_list[end]))_s($(seed_list[1])-$(seed_list[end]))_a($(ancilla_list[1])-$(ancilla_list[end])).jls"
    # filename = "MPS_(0,1)_L$(L)_p$(round(p, digits=2))_s$(seed).jls"
    open(filename, "w") do f
        serialize(f, Dict("O" => O_map, "EE" => EE_map, "max_bond" => max_bond_map, "L_list" => L_list, "p_list" => p_list, "seed_list" => seed_list, "ancilla_list" => ancilla_list))
    end

end

if isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# isdefined(Main, :PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__ && main()

# julia --sysimage ~/.julia/sysimages/sys_itensors.so run_CT_MPS.jl --p 1 --L 8 --seed 0 --ancilla 0