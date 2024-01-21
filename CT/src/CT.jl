module CT

using ITensors
using Random
using LinearAlgebra

# using TimerOutputs
# const to = TimerOutput()


mutable struct CT_MPS
    L::Int
    store_vec::Bool
    store_op::Bool
    store_prob::Bool
    seed::Union{Int,Nothing}
    seed_vec::Union{Int,Nothing}
    seed_C::Union{Int,Nothing}
    x0::Union{Rational{Int},Nothing}
    xj::Set
    _eps::Float64
    ancilla::Int
    folded::Bool
    rng::Random.AbstractRNG
    rng_vec::Random.AbstractRNG
    rng_C::Random.AbstractRNG
    qubit_site::Vector{Index{Int64}}
    phy_ram::Vector{Int}
    ram_phy::Vector{Int}
    phy_list::Vector{Int}
    _maxdim::Int
    _cutoff::Float64
    mps::MPS
    vec_history::Vector{MPS}
    op_history::Vector{Vector{Any}}
    debug::Bool
end

function CT_MPS(; L::Int, store_vec::Bool=false, store_op::Bool=false, store_prob::Bool=false, seed::Union{Nothing,Int}=nothing, seed_vec::Union{Nothing,Int}=nothing, seed_C::Union{Nothing,Int}=nothing, x0::Union{Rational{Int},Nothing}=nothing, xj::Set=Set([1 // 3, 2 // 3]), _eps::Float64=1e-10, ancilla::Int=0, folded::Bool=false, _maxdim::Int=10, _cutoff::Float64=1e-10,debug::Bool=false)
    rng = MersenneTwister(seed)
    rng_vec = seed_vec === nothing ? rng : MersenneTwister(seed_vec)
    rng_C = seed_C === nothing ? rng : MersenneTwister(seed_C)
    qubit_site, ram_phy, phy_ram, phy_list = _initialize_basis(L,ancilla,folded)
    mps=_initialize_vector(L,ancilla,x0,folded,qubit_site,ram_phy,phy_ram,phy_list,rng_vec,_cutoff,_maxdim)
    ct = CT_MPS(L, store_vec, store_op, store_prob, seed, seed_vec, seed_C, x0, xj, _eps, ancilla, folded, rng, rng_vec, rng_C, qubit_site, phy_ram, ram_phy, phy_list, _maxdim, _cutoff, mps, [],[],debug)
    return ct
end

function _initialize_basis(L,ancilla,folded)

    qubit_site = siteinds("Qubit", L+ancilla) # RAM site index
    # ram_phy[actual in ram] = physical 
    if ancilla ==0
        ram_phy = folded ? [i for pairs in zip(1:(L÷2), reverse((L÷2+1):L)) for i in pairs] : collect(1:L)
    elseif ancilla ==1
        ram_phy = folded ? vcat(L+1,[i for pairs in zip(1:(L÷2), reverse((L÷2+1):L)) for i in pairs]) : vcat(collect(1:L),L+1)
        # if folded : [L+1, 1,L, ..] else [1,2,..,L,L+1]
    elseif ancilla ==2
        error("Not implemented yet")
    end

    # phy_ram[physical] = actual in ram
    phy_ram = fill(0, L+ancilla)
    for (ram, phy) in enumerate(ram_phy)
        phy_ram[phy] = ram
    end

    phy_list = Dict(0 => 1:L,1=>1:L)[ancilla]
    return qubit_site, ram_phy, phy_ram, phy_list
end

function _initialize_vector(L,ancilla,x0,folded,qubit_site,ram_phy,phy_ram,phy_list,rng_vec,_cutoff,_maxdim)
    if ancilla == 0
        if x0 !== nothing
            vec_int = dec2bin(x0, L)
            vec_int_pos = [string(s) for s in lpad(string(vec_int, base=2), L, "0")] # physical index
            return MPS(qubit_site, [vec_int_pos[ram_phy[i]] for i in 1:L])
        else
            return randomMPS(rng_vec, qubit_site, linkdims=_maxdim)
        end
    elseif ancilla ==1
        if x0 !== nothing
            # This branch is only for debug purpose
            vec_int = dec2bin(x0, L)
            vec_int_pos = [string(s) for s in lpad(string(vec_int, base=2), L, "0")] # physical index
            push!(vec_int_pos,"0")
            return MPS(qubit_site, [vec_int_pos[ram_phy[i]] for i in 1:(L+ancilla)])
        else
            # Create two orthogonal MPS and couple them to |0> and |1> ancilla, respectively
            # The physical qubit of ancilla is always the last one
            # The ram position, depending on the folded or not, is either the first (folded) or the last (unfolded)
            
            mps1 = randomMPS(rng_vec, ComplexF64, qubit_site[sort(phy_ram[phy_list])], linkdims=_maxdim)
            mps0 = randomMPS(rng_vec, ComplexF64, qubit_site[sort(phy_ram[phy_list])], linkdims=_maxdim)
            mps1 -= inner(mps0, mps1) .* mps0
            mps1 = mps1 ./ norm(mps1)
            anc0 = MPS(qubit_site[[(phy_ram[L+1])]], ["0"])
            anc1 = MPS(qubit_site[[(phy_ram[L+1])]], ["1"])
            if folded
                return add(attach_mps(anc0,mps0) , attach_mps(anc1,mps1),cutoff=_cutoff)/sqrt(2)
            else 
                return (attach_mps(mps0,anc0) + attach_mps(mps1,anc1))/sqrt(2)
            end
        end
    end
    # return MPS(vec, ct.qubit_site; cutoff=ct._eps, maxdim=ct._maxdim)
end

"""
apply the operator `op` to `mps`, the `op` should have indices of (i,j,k.. i',j',k')
the orthogonalization center is at i
index should be ram index
"""
function apply_op!(mps::MPS, op::ITensor, cutoff::Float64)
    i_list = [parse(Int, replace(string(tags(inds(op)[i])[length(tags(inds(op)[i]))]), "n=" => "")) for i in 1:div(length(op.tensor.inds), 2)]
    sort!(i_list)
    # println(i_list)
    orthogonalize!(mps, i_list[1])
    mps_ij = mps[i_list[1]]
    for idx in i_list[1]+1:i_list[end]
        mps_ij *= mps[idx]
    end
    mps_ij *= op 
    noprime!(mps_ij)
    
    if length(i_list) == 1
        mps[i_list[1]] = mps_ij
    else
        lefttags= (i_list[1]==1) ? nothing : tags(linkind(mps,i_list[1]-1))
        for idx in i_list[1]:i_list[end]-1
            inds1 = (idx ==1) ? [siteind(mps,1)] : [findindex(mps[idx-1],lefttags), findindex(mps[idx],"Site")]
            lefttags=tags(linkind(mps,idx))
            U, S, V = svd(mps_ij, inds1, cutoff=cutoff,lefttags=lefttags)
            mps[idx] = U
            mps_ij = S * V
        end
        mps[i_list[end]] = mps_ij
    end
    return
end
""" apply Bernoulli_map to physical site i
"""
function Bernoulli_map!(ct::CT_MPS, i::Int)
    S!(ct, i, ct.rng_C)
end

""" apply scrambler (Haar random unitary) to site (i,i+1) [physical index]
"""
function S!(ct::CT_MPS, i::Int, rng::Random.AbstractRNG)
    # U=ITensor(1.)
    # U *= randomUnitary(linkind(mps,i), linkind(mps,i+1))
    # mps[i] *= U
    # mps[i+1] *= U
    # return
    U_4 = reshape(U(4,rng), 2, 2, 2, 2)
    if ct.ancilla == 0 || ct.ancilla ==1
        ram_idx = ct.phy_ram[[ct.phy_list[i], ct.phy_list[(i)%(ct.L)+1]]]
        U_4_tensor = ITensor(U_4, ct.qubit_site[ram_idx[1]], ct.qubit_site[ram_idx[2]], ct.qubit_site[ram_idx[1]]', ct.qubit_site[ram_idx[2]]')
        # return U_4_tensor
        apply_op!(ct.mps, U_4_tensor,ct._cutoff)
        if ct.debug
            println("U apply to $(i)")
        end
    elseif ct.ancilla ==2
        nothing
    end
end

""" apply control_map to physical site i
"""
function control_map(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    R!(ct, n, i)
    # normalize!(mps)
    # adder!(mps, n, i)
end

"""Rest: apply Projection to physical site list i with outcome list n, then reset to 1
"""
function R!(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    P!(ct, n, i)
    if ct.xj in [Set([1 // 3, 2 // 3]),Set([0])]
        if n[1]==1
            X!(ct,i[end])
            # print("X")
        end
    elseif ct.xj in [Set([1 // 3, -1 // 3])]
        if n[1]^n[2]==0
            X!(ct,i[end])
        end
    end
end
"""generate projector for physical site (will convert to RAM site)
"""
function projector(ct::CT_MPS,n::Vector{Int}, i::Vector{Int})
    if ct.debug
        println("Get projector $(n) at Phy $(i)")
    end
    proj_op=emptyITensor(ct.qubit_site[ct.phy_ram[ct.phy_list[i]]],ct.qubit_site[ct.phy_ram[ct.phy_list[i]]]')
    idx=n.+1
    proj_op[ idx...,idx... ]=1
    return proj_op
end
""" perform projection for physical site
"""
function P!(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    @assert length(n) == length(i) "length of n $(n) is not equal to length of i $(i)"
    if ct.debug
        println("Projecting $(n) at Phy $(i)")
    end
    proj_op= projector(ct,n, i)
    apply_op!(ct.mps, proj_op, ct._cutoff)
    if ct.debug
        println("norm is $(norm(ct.mps))")
    end
    normalize!(ct.mps)
    truncate!(ct.mps, cutoff=ct._cutoff)
end
""" perform Sigma_X for physical site
"""
function X!(ct::CT_MPS, i::Int)
    X_op=ITensor([0 1; 1 0],ct.qubit_site[ct.phy_ram[ct.phy_list[i]]],ct.qubit_site[ct.phy_ram[ct.phy_list[i]]]')
    apply_op!(ct.mps, X_op, ct._cutoff)
    # normalize!(ct.mps)
end


"""
compute the entanglement entropy of 1...i, i+1...L (ram sites)
"""
function von_Neumann_entropy(mps::MPS, i::Int)
    mps_ = orthogonalize(mps, i)
    # _,S = svd(mps_[i], (linkind(mps_, i-1), siteind(mps_,i)))
    _, S = svd(mps_[i], (linkind(mps_, i),))
    SvN = 0.0
    # print(S)
    for n = 1:dim(S, 1)
        p = S[n, n]^2
        SvN -= p * log(p)
    end
    return SvN
end

function mps_to_tensor(mps; array=false, vector=false, column_first=true)
    # Start with the first tensor
    psi = mps[1]

    # Sequentially contract the tensors
    for i = 2:length(mps)
        psi = psi * mps[i]
    end
    # Convert the resulting tensor to a dense array
    if array
        psi = array(psi)
        if vector
            if column_first
                return vec(psi)
            else
                return vec(permutedims(psi, reverse(1:ndims(psi))))
            end
        else
            return psi
        end
    else
        return psi
    end
end

# function encoding!(ct::CT_MPS)
#     i=1
#     for idx in 1:2*ct.L^2
#         Bernoulli_map!(ct, i)
#         # println(i)
#         i=mod(((i+1) - 1),ct.L+ct.ancilla )+ 1
#     end
# end
"""randomly apply control or Bernoulli map to physical site i
"""
function random_control!(ct::CT_MPS, i, p_ctrl, )
    
    op_l=[]
    p_0=-1.  # -1 for not applicable because of Bernoulli map
    if rand(ct.rng_C) < p_ctrl
        # control map
        if ct.xj in [Set([1 // 3, 2 // 3]),Set([0])]
            p_0= inner_prob(ct, [0], [i])
            if ct.debug
                println("Born prob for measuring 0 at phy site $i is $p_0")
            end
            n =  rand(ct.rng) < p_0 ?  0 : 1
            control_map(ct, [n], [i])
            push!(op_l,Dict("Type"=>"Control","Site"=>[i],"Outcome"=>[n]))
        elseif ct.xj in [Set([1 // 3, -1 // 3])]
            # p_00= ...
            # p_01= ...
            # p_10= ...
            # p_11= ...
            # n = ...
            # control_map(ct, n, [i, i+1])
            nothing
        end
        if ct.debug
            print("Control with $(i)")
        end
        i=mod(((i-1) - 1),(ct.L)) + 1
        if ct.debug
            println("=> Next i is $(i)")
        end
    else
        # chaotic map
        Bernoulli_map!(ct, i)
        push!(op_l,Dict("Type"=>"Bernoulli","Site"=>[i,((i+1) - 1)%(ct.L) + 1],"Outcome"=>nothing))
        i=mod(((i+1) - 1),(ct.L) )+ 1
    end
    update_history(ct,op_l,p_0)
    
    return i
end

function update_history(ct::CT_MPS,op::Vector{Any},p_0::Float64)
    if ct.store_vec
        push!(ct.vec_history,copy(ct.mps)) 
    end
    if ct.store_op
        push!(ct.op_history,op)
    end
    if ct.store_prob
        push!(ct.op_history,p_0)
    end
end
    

""" compute the Born probability through the inner product at physical site list i (will convert to RAM site internally)
"""
function inner_prob(ct::CT_MPS, n::Vector{Int}, i::Vector{Int})
    @assert length(n) == length(i) "length of n $(n) is not equal to length of i $(i)"
    ram_idx=ct.phy_ram[ct.phy_list[i]]
    if length(i)==1
        if ct.debug
            println("Get projector for inner_prob $(n) at Phy $(i) at RAM $(ram_idx)")
        end
        proj_op= array(projector(ct,n, i))
        return only(expect(ct.mps, proj_op,sites=ram_idx))
    else
        # proj_op= projector(ct,n, i,tensor=false)
        # os=OpSum()
        error("Not implemented yet")
    end

end

function order_parameter(ct::CT_MPS)
    if ct.xj in [Set([1 // 2, 2 // 3])]
        O = ZZ(ct)
    elseif ct.xj in [Set([0])]
        O = Z(ct)
    end
    return real(scalar(O))
end

function Zi(ct::CT_MPS)
    sZ = expect(ct.mps, "Sz")[ct.phy_ram][ct.phy_list]
    return sZ*2
end

function Z(ct::CT_MPS)
    sZ = Zi(ct)
    if ct.debug
        println("Z are $(2 * sZ)")
    end
    return sum(sZ) / ct.L
end

function ZZ(ct::CT_MPS)
    os = OpSum()
    for i in ct.phy_list[1:end-1]
        os += "Sz", ct.phy_ram[i], "Sz", ct.phy_ram[i+1]
    end
    os += "Sz", ct.phy_ram[ct.phy_list[end]], "Sz", ct.phy_ram[ct.phy_list[1]]
    zz = MPO(os, ct.qubit_site)
    # println(os)
    # println(zz)
    # println(ct.mps)
    return -inner(ct.mps', zz, ct.mps) * 4 / ct.L
end

function mps_element(mps::MPS, x::String)
    @assert length(x) == length(mps)
    x = [string(s) for s in x]
    V = ITensor(1.0)
    for i = 1:length(mps)
        V *= (mps[i] * state(siteind(mps, i), x[i]))
    end
    return scalar(V)
end



function dec2bin(x::Real, L::Int)
    @assert 0 <= x < 1 "$x is not in [0,1)"

    return BigInt(floor(x * (BigInt(1) << L)))
end

"""create Haar random unitary
"""
function U(n, rng::Random.AbstractRNG=MersenneTwister(nothing))
    z = randn(rng, 4, 4) + randn(rng, 4, 4) * im
    Q, R = qr(z)
    r_diag = diag(R)
    Lambda = Diagonal(r_diag ./ abs.(r_diag))
    Q *= Lambda
    return Q
end
"""attach mps2 to mps1 at the last site. The returned mps is |mps1>|mps2>
The physical index of mps2 should be given 
"""
function attach_mps(mps1::MPS,mps2::MPS,cutoff::Float64=1e-10;replace_link=true)
    n1=length(mps1)
    n2=length(mps2)
    if replace_link
        advance_link_tags!(mps2,n1)
    end
    orthogonalize!(mps1,n1)
    mps_=mps1[n1]*mps2[1]
    lefttags=replace(string(tags(findinds(inds(mps_),"Link")[1])),r"l=[\d]+"=>"l=$(n1)",'"'=>"")
    U,S,V=svd(mps_,inds(mps1[n1]),cutoff=cutoff,lefttags=lefttags)
    mps_new=MPS(vcat(siteinds(mps1),siteinds(mps2)))
    for i in 1:(n1+n2)
        if i<n1
            mps_new[i]=mps1[i]
        elseif i==n1
            mps_new[i]=U
        elseif i==n1+1
            mps_new[i]=S*V
        else
            mps_new[i]=mps2[i-n1]
        end
    end
    return mps_new
end
function advance_link_tags!(mps::MPS,n::Int64)
    for i in 1:length(mps)
        links=findinds(mps[i],"Link")
        for link in links
            cur_i=parse(Int,match(r"l=(\d+)",string(tags(link)[length(tags(link))]))[1])
            new_link=replacetags(link,"l=$(cur_i)","l=$(n+cur_i)")
            mps[i]*=delta(link,new_link)
        end
    end
end

function max_bond_dim(mps::MPS)
    max_dim = 0
    for i in 1:length(mps)-1
        dim = commonind(mps[i], mps[i+1])
        max_dim = max(max_dim, space(dim))
    end
    return max_dim
end


greet() = print("Hello World! H")


end # module CT
