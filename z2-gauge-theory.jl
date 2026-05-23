# * Initialization
using ITensors, ITensorMPS
ITensors.disable_warn_order();

using CSV
using DataFrames
using HDF5
using JSON3
using TimerOutputs
using Printf
using LinearAlgebra
using Random
using EasyFit

using Infiltrator

# Explicit BLAS thread setting. Default to half of physical threads,
# overridable via the Z2GT_BLAS_THREADS env var. Avoids oversubscription
# when ITensorMPS is already parallelized internally.
let n = parse(Int, get(ENV, "Z2GT_BLAS_THREADS", string(Sys.CPU_THREADS ÷ 2)))
    BLAS.set_num_threads(n)
    @info "Z2GT: BLAS threads set to $n"
end

# Plotting back-ends are heavy and unused by the Section III smoke run /
# phase-scan drivers. Try to load them but fall back to no-ops if absent.
const HAVE_MAKIE = try
    @eval using CairoMakie
    @eval using GLMakie
    @eval GLMakie.activate!()
    @eval using Makie.Colors
    true
catch err
    @warn "Makie/GLMakie/CairoMakie not available; plotting disabled" err
    false
end

pal = HAVE_MAKIE ? Makie.ColorSchemes.tab10.colors : nothing

# include("helper_functions.jl")
include(joinpath(pwd(), "helper_functions.jl"))
# include(joinpath(@__DIR__, "helper_functions.jl"))

module Data

@kwdef mutable struct Params
    m0 = 1.125;
    eta = 1.0;
    alpha = 1.0;

    a = 1;
    Lphys = 4;
    L = Int64(Lphys / a);

    ## alternating signs for charges
    qarr = [ i%2 ==1 ? -1 : 1 for i in 1:L ]

    #' Tensor network parameters
    truncate_negative_k = true
    dmrg_cutoff = 1e-11

    #' Output paths
    output_root = "../data/"

end

end

# * Strong coupling results
energy_strong_coupling_1_PBC(m0, alpha, L) = (-alpha/2)*L - m0*L/2;
meson_strong_coupling_1_PBC(m0, alpha, L) = +alpha + 2*m0;
glueball_strong_coupling_1_PBC(m0, alpha, L) = +(alpha) *L
Lcrit_strong_cpupling_1_PBC(m0, alpha) = 2*m0 / (1  + alpha)

# m0 >> alpha
energy_strong_coupling_1_OBC(m0, alpha, L) = (-alpha/2)*(L-1) - m0*L/2
energy_strong_coupling_1_PBC(m0, alpha, L) = (-alpha/2)*L - m0*L/2
# m0 << alpha
energy_strong_coupling_2_OBC(m0, alpha, L) = (alpha/2)*(L-1) - m0 * (L-4) / 2

function energy_strong_coupling(m0, alpha, L)
    if m0 >= alpha * (L - 1) / 2
        return energy_strong_coupling_1_OBC(m0, alpha, L)
    else
        return energy_strong_coupling_2_OBC(m0, alpha, L)
    end
end

###
# * Higgs = SPT with Energetic Gauss law

op_X_matter_string(n, offset) = Tuple(Iterators.flatten( (("X", i) for i in 1+offset:2:n) ))

function num_qubits(L, boundary)
    # Number of qubits is L+1, where L is the number of sites
    # OBC: N sites and (N-1) links
    if boundary == :open_link
        N = 2*L+1
    elseif boundary == :open_site
        N = 2*L-1
    elseif  boundary ==  :PBC
        N = 2*L
    end
    return N
end

function matter_site_offset(boundary)
    if boundary == :open_link
        return 1
    elseif boundary == :open_site
        return 0
    elseif  boundary ==  :PBC
        return 1
    else
        @warn "boundary not defined"
    end
end

function make_z2_higgs(p::Data.Params; boundary=:open_link)
    # qarr = p.qarr
    L = p.L
    alpha = p.alpha
    eta = p.eta
    m0 = p.m0

    # Number of qubits is L+1, where L is the number of sites
    # OBC: N sites and (N-1) links
    N = num_qubits(p.L, boundary)

    sites = siteinds("S=1/2", N; conserve_qns=false, conserve_szparity=false)

    os = OpSum()
    g = -1.0
    h = -1.0
    @show L

    if boundary == :open_site
        #' If boundary == :open_site, then we take j = 1 to be a site
        #' for all other cases, we take j = 1 to be a link


        for j in 1:L
            # (Site, Link) = (2j - 1, 2j) starts with (1,2)
            #' Gauss law terms: Link, Site, Link
            if 2j-1 <= 1
                # Gauss law on left boundary
                os += g, "X", 2j-1, "X", 2j
            elseif 2j-1 >= N
                # Gauss law on right boundary
                os += g, "X", 2j-2, "X", 2j-1
            else
                os += g, "X", 2j-2, "X", 2j-1, "X", 2j
            end

            #' Kinetic terms: (Site, Link, Site)
            if 2j+1 <= N
                os += h, "Z", 2j-1, "Z", 2j, "Z", 2j+1
            end

            os += -alpha, "X", 2j-1

            @show j
            @show os
        end
        
    else
        for j in 1:L
            # (Link, Site) = (2j - 1, 2j) starts with (1,2)
            os += -alpha, "X", 2j

            # Gauss law at first site
            # os += g, "X",

            #' If boundary == :open_site, then we take j = 1 to be a site
            #' for all other cases, we take j = 1 to be a link

            #' Gauss law terms: Link, Site, Link
            #' Start with (X1, X2, X3)
            #' End with (X_{2L-1}, X_{2L}, X_{2L+1})

            if 2j+1 <= N
                os += g, "X", 2j-1, "X", 2j, "X", 2j+1
            else # We need to worry about the boundary
                if boundary == :PBC
                    os += g, "X", 2j-1, "X", 2j, "X", (2j+1)%N 
                end
            end

            #' Kinetic terms: (Site, Link, Site)
            #' Start with (X2, X3, X4)
            #' End with (Z_{2L-2}, Z_{2L-1}, Z_{2L})
            #' (because 2L+1 is the last link, with boundary == :open_link)
            if 2j+2 <= N
                os += h, "Z", 2j, "Z", 2j+1, "Z", 2j+2
            else # We need to worry about this term going across the boundary
                #' Add the kinetic term for the last link for PBC
                #' (Z_{2L}, Z_{2L+1}, Z_{2L+2} ) == (Z_{2L}, Z_{1}, Z_{2} )
                if (boundary == :PBC)
                    os += h, "Z", 2j, "Z", (2j+1)%N, "Z", (2j+2)%N
                end
            end

            # @show j
            # @show os

        end

        # return H, sites
    end

    offset = matter_site_offset(boundary)
    # proj = OpSum()
    os += (m0, op_X_matter_string(N, offset)...)

    @show os
    # @exfiltrate
    
    H = MPO(os, sites)

    return H, sites
end

    
"Coefficient as a function of (say) site index j."
as_coeff(c::Function) = c
as_coeff(c) = (_ -> c)

struct OpTerm{Ops<:Tuple,F}
    coeff::F
    ops::Ops
end

OpTerm(c, ops...) = OpTerm(as_coeff(c), tuple(ops...))

"Turn various encodings into an iterator of (g, term::OpTerm) pairs."
pairs_from(x::OpTerm) = ((x.coeff, x),)
pairs_from(x::Tuple{Any,OpTerm}) = ((as_coeff(x[1]), x[2]),)
pairs_from(xs::AbstractVector) = Iterators.flatten(pairs_from.(xs))
function pairs_from(x)
    throw(ArgumentError("Unsupported term encoding: $(typeof(x)). Expected OpTerm, (g, OpTerm), or a Vector of those."))
end

"""
Apply terms stored under `key` in a dict.

`body!` is called as: body!(opsum, g, term)
and is expected to mutate `opsum` (or otherwise update it).
"""
function apply_terms!(body, opsum, data::AbstractDict, key)
    haskey(data, key) || return opsum
    for (g, term) in pairs_from(data[key])
        # @show g, term.ops
        add!(opsum, body(g, term.ops))
    end
    return nothing
end

# opsum_data = Dict(
#     # "site" => [OpTerm(1.0, "X")]
#     # also allowed:
#     # "site" => OpTerm(j -> 1.0*j, "Y"),
#     "link" => OpTerm(j -> 1.0*j, "Y"),
#     "site" => [
#         (j -> 2j, OpTerm(1.0, "X")),
#         (3.0, OpTerm(1.0, "Y"))
#     ]
# )

# opsum = OpSum()
# apply_terms!(opsum, opsum_data, "site") do g, term_ops
#     g(2), term_ops[1], 1
# end

# apply_terms!(opsum, opsum_data, "link") do g, term_ops
#     g(2), term_ops[1], 2
# end

# @show opsum


function make_lattice_gauge_opsum(opsum_data, L; boundary = :open_site)
    opsum = OpSum()

    N = num_qubits(L, boundary)

    if boundary == :open_site
        #' If boundary == :open_site, then we take j = 1 to be a site
        #' for all other cases, we take j = 1 to be a link

        for j in 1:L
            # (Site, Link) = (2j - 1, 2j) starts with (1,2)
            #' Gauss law terms: Link, Site, Link

            if 2j-1 <= 1
                # Gauss law on left boundary
                # @show "1"
                apply_terms!(opsum, opsum_data, "boundary_left") do g, ops
                    g(j), ops[1], 2j-1, ops[2], 2j
                end

            elseif 2j-1 >= N
                # Gauss law on right boundary
                # 
                # @show "2"
                apply_terms!(opsum, opsum_data, "boundary_right") do g, ops
                    g(j), ops[1], 2j-2, ops[2], 2j-1
                end
            else
                # @show "3"
                apply_terms!(opsum, opsum_data, "link-site-link") do g, ops
                    g(j), ops[1], 2j-2, ops[2], 2j-1, ops[3], 2j
                end
            end

            #' Kinetic terms: (Site, Link, Site)
            if 2j+1 <= N

                # @show "4"
                apply_terms!(opsum, opsum_data, "site-link-site") do g, ops
                    g(j), ops[1], 2j-1, ops[2], 2j, ops[3], 2j+1
                end
            end
            
                # @show "5"
            
            apply_terms!(opsum, opsum_data, "site") do g, ops
                g(j), ops[1], 2j-1
            end

        end
        
    else
        for j in 1:L
            # (Link, Site) = (2j - 1, 2j) starts with (1,2)
            # os += -alpha, "X", 2j
            # g, ops = opsum_data["site"]
            # opsum += g, ops[1], 2j

            apply_terms!(opsum, opsum_data, "site") do g, ops
                g(j), ops[1], 2j
            end

            # Gauss law at first site
            # os += g, "X",

            #' If boundary == :open_site, then we take j = 1 to be a site
            #' for all other cases, we take j = 1 to be a link

            #' Gauss law terms: Link, Site, Link
            #' Start with (X1, X2, X3)
            #' End with (X_{2L-1}, X_{2L}, X_{2L+1})

            # g, ops = opsum_data["link-site-link"]
            if 2j+1 <= N
                # os += g, "X", 2j-1, "X", 2j, "X", 2j+1
                apply_terms!(opsum, opsum_data, "link-site-link") do g, ops
                    g(j), ops[1], 2j-1, ops[2], 2j, ops[3], 2j+1
                end
            else # We need to worry about the boundary
                if boundary == :PBC
                    # os += g, "X", 2j-1, "X", 2j, "X", (2j+1)%N 
                    apply_terms!(opsum, opsum_data, "link-site-link") do g, ops
                        g(j), ops[1], 2j-1, ops[2], 2j, ops[3], (2j+1)%N 
                    end
                end
            end

            #' Kinetic terms: (Site, Link, Site)
            #' Start with (X2, X3, X4)
            #' End with (Z_{2L-2}, Z_{2L-1}, Z_{2L})
            #' (because 2L+1 is the last link, with boundary == :open_link)
            if 2j+2 <= N
                apply_terms!(opsum, opsum_data, "site-link-site") do g, ops
                    g(j), ops[1], 2j, ops[2], 2j+1, ops[3], 2j+2
                end
            else # We need to worry about this term going across the boundary
                #' Add the kinetic term for the last link for PBC
                #' (Z_{2L}, Z_{2L+1}, Z_{2L+2} ) == (Z_{2L}, Z_{1}, Z_{2} )
                if (boundary == :PBC)
                    # os += h, "Z", 2j, "Z", (2j+1)%N, "Z", (2j+2)%N
                    apply_terms!(opsum, opsum_data, "site-link-site") do g, ops
                        g(j), ops[1], 2j, ops[2], (2j+1)%N, ops[3], (2j+2)%N
                    end
                end
            end

            # @show j
            # @show opsum

        end

        # return H, sites
    end

    return opsum

end

function make_z2_gauge_theory_hamiltonian(p::Data.Params; boundary=:open_site,
                                          gauge_law::Symbol=:higgs_spt,
                                          lambda_gauss::Float64=20.0,
                                          higgs_obc_boundary::Symbol=:drop,
                                          z2_obc_boundary::Symbol=:drop,
                                          bg_left::Int=+1,
                                          bg_right::Int=+1)
    # qarr = p.qarr
    L = p.L
    alpha = p.alpha
    eta = p.eta
    m0 = p.m0

    g = -1.0
    h  = -1.0
    
    # Number of qubits is L+1, where L is the number of sites
    # OBC: N sites and (N-1) links
    N = num_qubits(p.L, boundary)

    sites = siteinds("S=1/2", N;
                     conserve_qns=false,
                     conserve_szparity=false)
    @show L

    # OBC boundary convention switch (applies only to gauge_law=:higgs_spt,
    # boundary=:open_site).
    #   :drop        -> no truncated boundary terms; canonical cluster model
    #                   with 2-fold-degenerate OBC ground state from edge modes.
    #   :truncate_xx -> include the truncated XX Gauss-law pieces at the two
    #                   OBC ends (X_{matter1}·X_{link1} and X_{linkN-1}·X_{matterN}).
    opsum_data_higgs_spt = if higgs_obc_boundary === :truncate_xx
        Dict(
            "site-link-site" => OpTerm(h, "Z", "Z", "Z"),
            "site" => OpTerm(-alpha, "X"),
            "boundary_left" => OpTerm(g, "X", "X"),
            "boundary_right" => OpTerm(g, "X", "X"),
            "link-site-link" => OpTerm(g, "X", "X", "X"),
        )
    else  # :drop
        Dict(
            "site-link-site" => OpTerm(h, "Z", "Z", "Z"),
            "site" => OpTerm(-alpha, "X"),
            "link-site-link" => OpTerm(g, "X", "X", "X"),
        )
    end

    opsum_data_z2 = Dict(
        # Hopping term (η/4)(XX+YY)σᶻ from paper Eq. 2.
        "site-link-site" => [
            OpTerm(p.eta/4, "X", "Z", "X"),
            OpTerm(p.eta/4, "Y", "Z", "Y"),
        ],

        # Staggered fermion mass -(m₀/2)(-1)^n Z_n.
        "site" => OpTerm(j -> -p.m0/2 * (j%2==0 ? 1 : -1), "Z"),

        # Gauss-law SOFT PENALTY for the Z₂-LGT.
        # G_n = σˣ_{n-1,n} · Z_n · σˣ_{n,n+1}, background charge η_n = (-1)^n.
        # Add -(λ/2) η_n G_n at every interior matter site; constant from
        # (λ/2)(1-η_n G_n) is dropped (global offset, doesn't affect gaps).
        # Boundary generators are dropped at OBC ends by default (only
        # "link-site-link" key is set). When z2_obc_boundary=:truncate_xz
        # below, we additionally activate "boundary_left" / "boundary_right".
        "link-site-link" => OpTerm(j -> -lambda_gauss/2 * ((-1)^j), "X", "Z", "X"),
    )

    # If user asked for the truncated Gauss-law version of OBC, add the
    # 2-qubit boundary penalty terms. These are only meaningful for
    # boundary=:open_site (PBC has no missing-link issue, :open_link uses a
    # different layout).
    #
    # Left boundary: matter at qubit 1, right link at qubit 2.
    #   G_1 = (+1)·Z_1·σˣ_{1,2} = Z·X on (1,2).
    #   With η_1 = (-1)^1 = -1, penalty is -(λ/2)·η_1·G_1 = +(λ/2)·Z·X.
    # Right boundary: matter at qubit 2L-1, left link at qubit 2L-2.
    #   G_L = σˣ_{L-1,L}·Z_L·(+1) = X·Z on (2L-2, 2L-1).
    #   η_L = (-1)^L, penalty is -(λ/2)·η_L·G_L = -(λ/2)·(-1)^L · X·Z.
    if gauge_law == :z2 && boundary == :open_site && z2_obc_boundary == :truncate_xz
        # Penalty -(λ/2)·η_n·bg_n·G_n. η_1 = -1, η_L = (-1)^L.
        # Default bg=+1 at both ends reproduces the prior code.
        opsum_data_z2["boundary_left"]  = OpTerm(j -> +lambda_gauss/2 * bg_left, "Z", "X")
        opsum_data_z2["boundary_right"] = OpTerm(j -> -lambda_gauss/2 * ((-1)^j) * bg_right, "X", "Z")
    end

    # Select gauge-law form. Default preserves prior behaviour (:higgs_spt).
    # gauge_law=:z2 selects the Eq.2 form of the paper (XZX+YZY hopping,
    # staggered Z mass).
    opsum_data = gauge_law == :z2 ? opsum_data_z2 : opsum_data_higgs_spt
    opsum = make_lattice_gauge_opsum(opsum_data, L; boundary)

    # For the Eq.2 form, add the electric field energy on links:
    #   (alpha/2) * sum_links sigma^x_link   (alpha=1 reproduces Eq.2).
    # Link qubits in our enumeration depend on the boundary mode:
    #   :open_site  -> links live at even indices 2,4,...,2L-2  (L-1 links)
    #   :open_link  -> links live at odd indices  1,3,...,2L+1  (L+1 links)
    #   :PBC        -> links live at odd indices  1,3,...,2L-1  (L links, wraps)
    if gauge_law == :z2
        coeff_E = alpha / 2
        if boundary == :open_site
            for k in 1:(L-1)
                opsum += coeff_E, "X", 2k
            end
        elseif boundary == :open_link
            for k in 0:L
                opsum += coeff_E, "X", 2k+1
            end
        elseif boundary == :PBC
            for k in 0:(L-1)
                opsum += coeff_E, "X", 2k+1
            end
        end
    end

    # offset = matter_site_offset(boundary)
    # # proj = OpSum()
    # os += (m0, op_X_matter_string(N, offset)...)
    # @show os
    # @exfiltrate
    
    @show opsum

    H = MPO(opsum, sites)

    return H, sites
end

# NOTE: previous top-level smoke calls were moved into `main()` further down,
# behind an `if abspath(PROGRAM_FILE) == @__FILE__` guard, so that
# `include("z2-gauge-theory.jl")` no longer auto-runs DMRG.
#
# Original calls kept for reference:
#     m0 = 1.125; eta = 1.0; alpha = 1.0; L = 6
#     p = Data.Params(L = L; m0 = 0.0, alpha = 0.1)
#     H, sites = make_z2_higgs(p; boundary = :open_site)
#     H, sites = make_z2_gauge_theory_hamiltonian(p; boundary = :open_site)

###

function solve_z2_higgs(H, sites)
    N = length(sites)
    # sites = siteinds("S=1/2", N;
    #                  conserve_qns = false,
    #                  conserve_szparity = false,)

    psi0 = random_mps(sites;linkdims=2)

    nsweeps = 50
    maxdim = [5, 10, 20, 100, 100, 200, 400, 800]
    cutoff = 1.0e-10
    noise = [1e-8]
    weight = 10

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized MPS
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
    println("Final energy = $energy")

    # psi1_init = random_mps(sites; linkdims=2)
    psi1_init = psi0
    energy1,psi1 = dmrg(H, [psi], psi1_init; nsweeps, maxdim, cutoff, noise, weight)

    # psi2_init = random_mps(sites; linkdims=2)
    psi2_init = psi0
    energy2, psi2 = dmrg(H, [psi, psi1], psi2_init;
                        nsweeps, maxdim, cutoff, noise, weight)

    @show gap1 = energy1 - energy
    @show gap2 = energy2 - energy

    Zval = expect(psi, "Z")
    energies = [energy, energy1, energy2]
    psi_list = [psi, psi1, psi2]
    # return energy, Zval[N÷2]
    return  energies, psi_list
end

function solve_exact_z2_higgs(H, sites )
    # if L >= 10
    #     @warn ("L too big", L)
    #     return 0.0, 0.0, zeros(L)
    # end

    hmat = mpo_to_array(H, sites)
    eig_hmat = eigen(Hermitian(hmat))
    e0 = eig_hmat.values[1]
    e1 = eig_hmat.values[2]
    @show gap = e1 - e0

    return eig_hmat
end

function parity_op_X(sites, p::Data.Params; boundary)
    ## Construct  P = 1 - (Z_1...Z_N) (q_1...q_N)
    @show boundary

    N = num_qubits(p.L, boundary)
    offset = matter_site_offset(boundary)
    proj = OpSum()
    proj += (1.0, op_X_matter_string(N, offset)...)
    # proj  += 0.5,"Id",1

    # @show N
    # @show offset
    # @show proj
    parity_mpo = MPO(proj, sites)
    # proj_mat = mpo_to_array(proj_mpo, sites)
    @exfiltrate

    return parity_mpo
end

# p = Data.Params(L = 4; m0 = 0.0, alpha = 0.1)
# bdy = :open_site;
# # bdy = :open_link;
# # bdy = :PBC;
# H, sites = make_z2_higgs(p; boundary = bdy) # 

# Hmat = mpo_to_array(H, sites);
# esys = solve_exact_z2_higgs(H, sites);
# evals = round.(esys.values ; digits = 10);
# evals_uniq = unique(x -> round(x; digits = 10), evals);
# @show gap = evals_uniq[2] - evals_uniq[1];
# @show gap = evals_uniq[3] - evals_uniq[1];

# op_X_matter_string(n, offset) = Tuple(Iterators.flatten( (("X", i) for i in 1+offset:2:num_qubits()) ))

# state0 = esys.vectors[:, 1];

# Pmpo = parity_op_X(sites, p; boundary = bdy);
# Pmat = mpo_to_array(Pmpo, sites);

# Pval(i) = esys.vectors[:, i]' * Pmat * esys.vectors[:, i]

# Pvals = round.(Pval.(1:10); digits = 4)
# @show Pvals
# @show evals[1:10]

# does_commute(A, B) = norm(A*B - B*A);

# df = DataFrame(:E => evals[1:10], :P => Pvals);
# df.E = df.E - p.m0 .* df.P;
# display(df)

# gap = e1 - e0 = 1.800062280838345
# gap = e1 - e0 = 1.867106724670403

function check_L_z2_higgs(p; boundary = :PBC, )
    #=
    if false 
        Hmat = mpo_to_array(H, sites);
        esys = solve_exact_z2_higgs(H, sites);
        evals = round.(esys.values ; digits = 10);
        evals_uniq = unique(x -> round(x; digits = 10), evals);
        @show gap = evals_uniq[2] - evals_uniq[1];
        @show gap = evals_uniq[3] - evals_uniq[1];

        # op_X_matter_string(n, offset) = Tuple(Iterators.flatten( (("X", i) for i in 1+offset:2:num_qubits()) ))
        op_X_matter_string(n, offset) = Tuple(Iterators.flatten( (("X", i) for i in 1+offset:2:n) ))

        state0 = esys.vectors[:, 1];

        Pmpo = parity_op_X(sites, p; boundary = bdy);
        Pmat = mpo_to_array(Pmpo, sites);

        Pval(i) = esys.vectors[:, i]' * Pmat * esys.vectors[:, i]

        Pvals = round.(Pval.(1:10); digits = 4)
        @show Pvals
        @show evals[1:10]

        does_commute(A, B) = norm(A*B - B*A);

        df = DataFrame(:E => evals[1:10], :P => Pvals);
        df.E = df.E - p.m0 .* df.P;
        display(df)
    end
    =#


    L = p.L
    @show L
    bdy = boundary
    # bdy = :open_site;
    # bdy = :open_link;
    # bdy = :PBC;
    # H, sites = make_z2_higgs(p; boundary = bdy) # 
    H, sites = make_z2_gauge_theory_hamiltonian(p; boundary = bdy) # 
    energies, psis = solve_z2_higgs(H, sites)

    # @show (spec_obc.-spec_obc[1])[1:10]
    # @show (spec_pbc.-spec_pbc[1])[1:10]
    # @show gap_obc = spec_obc[3] - spec_obc[1]
    # @show gap_pbc = spec_pbc[3] - spec_pbc[1]

    return energies[2] - energies[1]

    # @exfiltrate
end

function get_z2_massgap_plots()

    function check_L_z2(p; boundary = :open_site, old=true)
        bdy = boundary

        if old
            H, sites = make_z2_higgs(p; boundary = bdy) #
        else
            H, sites = make_z2_gauge_theory_hamiltonian(p; boundary = bdy) # 
        end
            
        energies, psis = solve_z2_higgs(H, sites)
        return energies[2] - energies[1]
    end

    df = DataFrame();

    Larr = [4]
    # Larr = [4]
    for L in Larr
        @show L
        local p = Data.Params(L = L; m0 = 0.0, alpha = 0.1)

        # H, sites = make_z2_higgs(p; boundary = bdy) #
        # H, sites = make_z2_gauge_theory_hamiltonian(p; boundary = bdy) # 

        old = true
        @show L
        @show gap_obc_site = check_L_z2(p, boundary = :open_site, old = old )
        @show gap_pbc = check_L_z2(p, boundary = :PBC, old = old)
        push!( df, (;  :L=> L, :gap_obc_site => gap_obc_site, :gap_pbc => gap_pbc) )

        old = false
        @show gap_obc_site = check_L_z2(p, boundary = :open_site, old = old )
        @show gap_pbc = check_L_z2(p, boundary = :PBC, old = old)

        # gap = check_L_z2_higgs(L)
        push!( df, (;  :L=> L, :gap_obc_site => gap_obc_site, :gap_pbc => gap_pbc) )

    end

    @exfiltrate
    display(df)

    # fig = Figure();
    # ax = Axis(fig[1, 1]);

    # scatterlines!(df.L, df.gap_pbc; label = "PBC" )

    # # fit = fitexp(df.L[3:end], df.gap_obc_site[3:end])

    # # LL = 4:0.01:20
    # scatterlines!(df.L, df.gap_obc_site; linestyle=:dash, label = "OBC", color = pal[2] )
    # # lines!(LL, fit.(LL); label = @sprintf("fit, gap = %.3f", fit.c), color = pal[2])
    # # ax.title = @sprintf("m0 = %.2f, η = %.2f, α = %.2f", m0, eta, alpha)
    
    # display(df)

    # axislegend(ax)
    # display(fig)

    # save("output/gap_higgs.pdf", fig)

    return nothing
    # return fig
end

# (Previous unguarded driver call moved into `main()` below.)
# fig = get_z2_massgap_plots()


# for L in [4, 6]
#     push!(df, (; L,
#                 e0_pbc,
#                 e1_pbc,
#                 e2_pbc,
#                 e0_obc,
#                 e1_obc,
#                 e2_obc,
#                 e0_pbc_exact,
#                 e1_pbc_exact,
#                 e2_pbc_exact,
#                 e0_obc_exact,
#                 e1_obc_exact,
#                 e2_obc_exact,
#                 ); promote=true)

###

# ============================================================================
# * SPT / topological-phase diagnostics
# ----------------------------------------------------------------------------
# These work on an MPS `psi` (with `sites = siteinds(psi)`) regardless of which
# Hamiltonian / gauge-law choice produced it. They are tuned for finite chains
# in the 1+1d Z2 lattice gauge theory but are generic enough to be informative
# on any spin-1/2 chain.
# ============================================================================

"""
    entanglement_spectrum(psi, bond) -> Vector{Float64}

Schmidt values (sorted descending) of `psi` across the bond between sites
`bond` and `bond+1`. The Schmidt values squared sum to 1. Truncated values
below 1e-14 are dropped.
"""
function entanglement_spectrum(psi::MPS, bond::Integer)
    N = length(psi)
    @assert 1 <= bond < N "bond must be in 1:N-1"
    psi2 = orthogonalize(copy(psi), bond)
    # SVD the site tensor at `bond`, separating left index + site index
    # from the right link.
    A = psi2[bond]
    # Left indices: previous link (if any) + site index at `bond`.
    li = bond > 1 ? linkind(psi2, bond - 1) : nothing
    si = siteind(psi2, bond)
    left_inds = li === nothing ? (si,) : (li, si)
    U, S, V = svd(A, left_inds...)
    # `S` is a diagonal 2-index ITensor (u, v). Pull out its diagonal.
    Sarr = Array(S, inds(S)...)
    n = min(size(Sarr)...)
    svals = [Sarr[k, k] for k in 1:n]
    svals = filter(s -> s > 1e-14, real.(svals))
    return sort(svals; rev=true)
end

"""
    es_degeneracy_ratio(svals; tol=1e-3) -> Float64

Fraction of Schmidt values that come in (near-)degenerate pairs.

A pair (s_i, s_{i+1}) is counted as degenerate when
    |s_i - s_{i+1}| / max(s_i, 1e-12) < tol
Each matched pair contributes 2 values to the count. The result is in [0,1].

In the canonical Z2xZ2 SPT phase (Haldane-like), every Schmidt value is doubly
degenerate so this ratio approaches 1. A trivial phase typically has many
non-degenerate values so the ratio is small.

Note: threshold `tol=1e-3` is a heuristic and the cross-check agent may
reasonably pick something different.
"""
function es_degeneracy_ratio(svals::AbstractVector{<:Real}; tol::Real=1e-3)
    n = length(svals)
    n < 2 && return 0.0
    paired = falses(n)
    i = 1
    while i < n
        if !paired[i]
            s = svals[i]
            if abs(svals[i+1] - s) / max(s, 1e-12) < tol
                paired[i] = true
                paired[i+1] = true
                i += 2
                continue
            end
        end
        i += 1
    end
    return count(paired) / n
end

"""
    entanglement_entropy_profile(psi) -> Vector{Float64}

von Neumann entanglement entropy S(x) = -sum p_i log(p_i), p_i = s_i^2,
across every bond x ∈ 1:N-1 of an N-site MPS.
"""
function entanglement_entropy_profile(psi::MPS)
    N = length(psi)
    S = zeros(Float64, N - 1)
    for x in 1:(N - 1)
        svals = entanglement_spectrum(psi, x)
        p = svals .^ 2
        p = p ./ sum(p)
        S[x] = -sum(pi -> pi > 0 ? pi * log(pi) : 0.0, p)
    end
    return S
end

"""
    edge_entropy(psi; side=:left) -> Float64

Single-site von Neumann entropy of the leftmost (or rightmost) site reduced
density matrix. Computed as the entanglement entropy across the bond between
site 1 and site 2 (or between site N-1 and site N), which equals the single-
site entropy of site 1 (or N) because the MPS is pure.
"""
function edge_entropy(psi::MPS; side::Symbol=:left)
    N = length(psi)
    bond = side == :left ? 1 : N - 1
    svals = entanglement_spectrum(psi, bond)
    p = svals .^ 2
    p ./= sum(p)
    return -sum(pi -> pi > 0 ? pi * log(pi) : 0.0, p)
end

"""
    string_order(psi, sites, op_end::String, op_mid::String, i::Int, j::Int) -> Float64

Compute <psi | O_i^{end} * (prod_{k=i+1}^{j-1} O_k^{mid}) * O_j^{end} | psi>.
Generic builder used by the symmetry-specific string orders below.
"""
function string_order(psi::MPS, sites, op_end::String, op_mid::String, i::Int, j::Int)
    @assert 1 <= i < j <= length(sites)
    os = OpSum()
    # Build the product term-by-term. Use a single n-body operator string.
    ops_tuple = Any[]
    push!(ops_tuple, op_end); push!(ops_tuple, i)
    for k in (i+1):(j-1)
        push!(ops_tuple, op_mid); push!(ops_tuple, k)
    end
    push!(ops_tuple, op_end); push!(ops_tuple, j)
    os += (1.0, ops_tuple...)
    mpo = MPO(os, sites)
    val = inner(psi', mpo, psi)
    return real(val)
end

"""
    string_order_z2_higgs_spt(psi, sites) -> NamedTuple

Z2 x Z2 SPT-style string order parameters for the Higgs/SPT-form
Hamiltonian (XXX gauge-law, ZZZ kinetic, on-site X mass).

We measure the two standard SPT strings of opposite parities and return
both. They take the form
   O_a = <Z_i * (X_{i+1} X_{i+2} ... X_{j-1}) * Z_j>
   O_b = <X_i * (Z_{i+1} Z_{i+2} ... Z_{j-1}) * X_j>
with i and j taken near the boundaries (avoiding endpoints).

CAVEAT: the natural "string" for a Z2 gauge theory in this representation
is not unique. We pick the spin-chain Z2 x Z2 strings because they are
diagnostic of the SPT phase known to be present in this model; the choice
is documented here so the cross-check agent can match it.
"""
function string_order_z2_higgs_spt(psi::MPS, sites)
    N = length(sites)
    # Use a string from the second site to the second-to-last site.
    i = 2; j = N - 1
    O_zxz = N >= 4 ? string_order(psi, sites, "Z", "X", i, j) : NaN
    O_xzx = N >= 4 ? string_order(psi, sites, "X", "Z", i, j) : NaN
    return (O_zxz=O_zxz, O_xzx=O_xzx)
end

"""
    string_order_z2_eq2(psi, sites) -> NamedTuple

String order parameter appropriate for the Eq.2 Hamiltonian
(XZX+YZY hopping, staggered Z mass). The natural symmetries here are:
  - fermion parity P_F = prod_{n in matter} Z_n  (after JW)
  - the Z2 Gauss law on each link.
We measure the fermion-parity string

    O_F = <Z_{n0} * (Z_{n0+2} * Z_{n0+4} * ... * Z_{n1-2}) * Z_{n1}>

taken over matter sites only. We also return the "mesonic" XZX-like
two-point function used in the Eq.2 hopping term.

CAVEAT: as for the Higgs form, the operator choice is not unique.
Documented for cross-check.
"""
function string_order_z2_eq2(psi::MPS, sites, boundary::Symbol)
    N = length(sites)
    # Matter-site indices (1-based)
    offset = matter_site_offset(boundary)
    matter_idx = collect((1 + offset):2:N)
    length(matter_idx) < 3 && return (O_F=NaN, O_meson=NaN)
    i = matter_idx[1]
    j = matter_idx[end]
    # Fermion-parity string: Z at every matter site between i and j.
    # Build manually so it skips link qubits.
    os = OpSum()
    ops_tuple = Any[]
    for n in matter_idx
        push!(ops_tuple, "Z"); push!(ops_tuple, n)
    end
    os += (1.0, ops_tuple...)
    mpo = MPO(os, sites)
    O_F = real(inner(psi', mpo, psi))
    # Mesonic 2-point: X_{i} Z_{link between i and j} X_{j} with the
    # *adjacent* matter pair (i and i+2 are nearest matter neighbours, with a
    # single link qubit between them at i+1).
    if length(matter_idx) >= 2
        a = matter_idx[1]
        b = matter_idx[2]
        link = (a + b) ÷ 2
        os2 = OpSum()
        os2 += 1.0, "X", a, "Z", link, "X", b
        mpo2 = MPO(os2, sites)
        O_meson = real(inner(psi', mpo2, psi))
    else
        O_meson = NaN
    end
    return (O_F=O_F, O_meson=O_meson)
end

"""
    spt_diagnostics(psi, sites; gauge_law::Symbol, boundary::Symbol) -> NamedTuple

Bundle of diagnostics computed on a single MPS:
  - es: entanglement spectrum at the central bond
  - es_deg: degeneracy ratio of `es`
  - S_profile: entanglement entropy profile across all bonds
  - S_edge_L, S_edge_R: edge entropies
  - string_order: NamedTuple of string order parameters (form depends on `gauge_law`)
"""
function spt_diagnostics(psi::MPS, sites; gauge_law::Symbol, boundary::Symbol)
    N = length(sites)
    bond_mid = max(1, N ÷ 2)
    es = entanglement_spectrum(psi, bond_mid)
    deg = es_degeneracy_ratio(es)
    S_prof = entanglement_entropy_profile(psi)
    S_L = edge_entropy(psi; side=:left)
    S_R = edge_entropy(psi; side=:right)
    so = if gauge_law == :z2
        string_order_z2_eq2(psi, sites, boundary)
    else
        string_order_z2_higgs_spt(psi, sites)
    end
    return (es=es, es_deg=deg, S_profile=S_prof,
            S_edge_L=S_L, S_edge_R=S_R, string_order=so)
end

"""
    classify_phase(diag; gap, gap_tol=0.05, deg_tol=0.7, so_tol=0.1) -> Symbol

Heuristic phase classification from a `spt_diagnostics` named tuple plus
an externally-supplied energy gap `gap` (in units of the Hamiltonian).

Returns one of:
  - :critical          (gap < gap_tol)
  - :spt               (gap >= gap_tol, ES degeneracy ratio >= deg_tol,
                        and at least one string order has |O| >= so_tol)
  - :topological       (gap >= gap_tol, large ES degeneracy, but small
                        string order — could indicate intrinsic topological
                        order rather than SPT)
  - :trivial_gapped    (otherwise)

CAVEATS:
  - gap_tol = 0.05 is in absolute energy units. For very small L the gap can
    look large even in a critical phase.
  - deg_tol = 0.7: at least 70% of Schmidt values must be paired.
  - so_tol = 0.1: arbitrary; cross-check agent may use a different number.
"""
function classify_phase(diag; gap::Real,
                        gap_tol::Real=0.05,
                        deg_tol::Real=0.7,
                        so_tol::Real=0.1)
    if abs(gap) < gap_tol
        return :critical
    end
    so = diag.string_order
    so_vals = collect(values(so))
    so_max = maximum(abs.(filter(!isnan, so_vals)); init=0.0)
    is_spt_like = diag.es_deg >= deg_tol
    if is_spt_like && so_max >= so_tol
        return :spt
    elseif is_spt_like
        return :topological
    else
        return :trivial_gapped
    end
end

# ============================================================================
# * Section III drivers (paper Sec III \itodo items)
# ============================================================================

"""
    build_H(p; boundary, gauge_law) -> (H, sites)

Convenience wrapper that returns the Hamiltonian for either gauge-law choice.
`gauge_law` ∈ {:higgs_spt, :z2}.

For `:higgs_spt` we use `make_z2_gauge_theory_hamiltonian(...; gauge_law=:higgs_spt)`,
which is the dispatcher form of the older `make_z2_higgs`. The two builders
differ at the boundary so we do NOT mix them within one cross-check.
"""
function build_H(p::Data.Params; boundary::Symbol, gauge_law::Symbol,
                 higgs_obc_boundary::Symbol=:drop,
                 z2_obc_boundary::Symbol=:drop,
                 bg_left::Int=+1, bg_right::Int=+1)
    if gauge_law == :higgs_spt_old
        return make_z2_higgs(p; boundary=boundary)
    else
        return make_z2_gauge_theory_hamiltonian(p; boundary=boundary,
                                                gauge_law=gauge_law,
                                                higgs_obc_boundary=higgs_obc_boundary,
                                                z2_obc_boundary=z2_obc_boundary,
                                                bg_left=bg_left,
                                                bg_right=bg_right)
    end
end

"""
    section_iii_smoke_run() -> DataFrame

Runs the paper's Sec III smoke checks at the reference point
m_0 = 1.125, η = 1.0, α = 1.0.

For each L in {4, 6, 8} and each gauge_law in {:higgs_spt, :z2} computes:
  - gap_OBC (boundary = :open_site)
  - gap_PBC (boundary = :PBC)
  - ratio gap_OBC / gap_PBC
  - SPT diagnostics + classification at OBC ground state.

Writes the result to data/z2_gauge_theory/section_iii_smoke.csv .
"""
function section_iii_smoke_run(; Larr=[4, 6, 8],
                               gauge_laws=[:higgs_spt, :z2],
                               higgs_obc_boundary::Symbol=:drop,
                               outdir=joinpath(@__DIR__, "data", "z2_gauge_theory"))
    mkpath(outdir)
    @info "section_iii_smoke_run" higgs_obc_boundary
    df = DataFrame(L=Int[], gauge_law=Symbol[],
                   gap_OBC=Float64[], gap_PBC=Float64[],
                   ratio_OBC_over_PBC=Float64[],
                   es_deg_OBC=Float64[],
                   S_edge_L_OBC=Float64[], S_edge_R_OBC=Float64[],
                   string_order_1_OBC=Float64[],
                   string_order_2_OBC=Float64[],
                   phase_OBC=Symbol[])

    for L in Larr, gl in gauge_laws
        @info "section_iii_smoke_run: L=$L, gauge_law=$gl"
        p = Data.Params(; m0=1.125, eta=1.0, alpha=1.0, Lphys=L, L=L)

        # OBC
        gap_obc = NaN; diag_obc = nothing
        try
            H_obc, sites_obc = build_H(p; boundary=:open_site, gauge_law=gl,
                                       higgs_obc_boundary=higgs_obc_boundary)
            es, psis = solve_z2_higgs(H_obc, sites_obc)
            gap_obc = es[2] - es[1]
            diag_obc = spt_diagnostics(psis[1], sites_obc;
                                       gauge_law=gl, boundary=:open_site)
        catch err
            @warn "OBC failed at L=$L, gauge_law=$gl" exception=(err, catch_backtrace())
        end

        # PBC
        gap_pbc = NaN
        try
            H_pbc, sites_pbc = build_H(p; boundary=:PBC, gauge_law=gl,
                                       higgs_obc_boundary=higgs_obc_boundary)
            es, _ = solve_z2_higgs(H_pbc, sites_pbc)
            gap_pbc = es[2] - es[1]
        catch err
            @warn "PBC failed at L=$L, gauge_law=$gl" exception=(err, catch_backtrace())
        end

        ratio = gap_pbc != 0 ? gap_obc / gap_pbc : NaN

        if diag_obc !== nothing
            phase = classify_phase(diag_obc; gap=gap_obc)
            so = diag_obc.string_order
            so_vals = collect(values(so))
            so_1 = length(so_vals) >= 1 ? so_vals[1] : NaN
            so_2 = length(so_vals) >= 2 ? so_vals[2] : NaN
            push!(df, (L=L, gauge_law=gl,
                       gap_OBC=gap_obc, gap_PBC=gap_pbc,
                       ratio_OBC_over_PBC=ratio,
                       es_deg_OBC=diag_obc.es_deg,
                       S_edge_L_OBC=diag_obc.S_edge_L,
                       S_edge_R_OBC=diag_obc.S_edge_R,
                       string_order_1_OBC=so_1,
                       string_order_2_OBC=so_2,
                       phase_OBC=phase))
        else
            push!(df, (L=L, gauge_law=gl,
                       gap_OBC=gap_obc, gap_PBC=gap_pbc,
                       ratio_OBC_over_PBC=ratio,
                       es_deg_OBC=NaN,
                       S_edge_L_OBC=NaN, S_edge_R_OBC=NaN,
                       string_order_1_OBC=NaN, string_order_2_OBC=NaN,
                       phase_OBC=:unknown))
        end
    end

    csvpath = joinpath(outdir, "section_iii_smoke_higgs_$(higgs_obc_boundary).csv")
    CSV.write(csvpath, df)
    @info "Wrote $csvpath"
    display(df)
    return df
end

"""
    phase_scan_smoke(; L=4, boundary=:open_site, gauge_law=:z2,
                     m0_grid=[0.0, 0.5, 1.0, 1.5],
                     eta_grid=[0.5, 1.0, 1.5]) -> DataFrame

Small (m_0, η) phase scan at fixed L and boundary. For each grid point
runs DMRG, computes SPT diagnostics, and classifies the phase.

Writes data/z2_gauge_theory/phase_scan.csv .
"""
function phase_scan_smoke(; L::Int=4,
                          boundary::Symbol=:open_site,
                          gauge_law::Symbol=:z2,
                          m0_grid=[0.0, 0.5, 1.0, 1.5],
                          eta_grid=[0.5, 1.0, 1.5],
                          alpha::Float64=1.0,
                          outdir=joinpath(@__DIR__, "data", "z2_gauge_theory"))
    mkpath(outdir)
    df = DataFrame(L=Int[], boundary=Symbol[], gauge_law=Symbol[],
                   m0=Float64[], eta=Float64[],
                   gap=Float64[], es_deg=Float64[],
                   S_edge_L=Float64[], S_edge_R=Float64[],
                   string_order_1=Float64[], string_order_2=Float64[],
                   phase=Symbol[])

    for m0 in m0_grid, eta in eta_grid
        @info "phase_scan_smoke: m0=$m0, eta=$eta (L=$L, $boundary, $gauge_law)"
        try
            p = Data.Params(; m0=m0, eta=eta, alpha=alpha, Lphys=L, L=L)
            H, sites = build_H(p; boundary=boundary, gauge_law=gauge_law)
            es, psis = solve_z2_higgs(H, sites)
            gap = es[2] - es[1]
            diag = spt_diagnostics(psis[1], sites;
                                   gauge_law=gauge_law, boundary=boundary)
            phase = classify_phase(diag; gap=gap)
            so_vals = collect(values(diag.string_order))
            push!(df, (L=L, boundary=boundary, gauge_law=gauge_law,
                       m0=m0, eta=eta, gap=gap,
                       es_deg=diag.es_deg,
                       S_edge_L=diag.S_edge_L, S_edge_R=diag.S_edge_R,
                       string_order_1=so_vals[1],
                       string_order_2=length(so_vals) >= 2 ? so_vals[2] : NaN,
                       phase=phase))
        catch err
            @warn "phase_scan_smoke point failed" m0 eta exception=(err, catch_backtrace())
            push!(df, (L=L, boundary=boundary, gauge_law=gauge_law,
                       m0=m0, eta=eta, gap=NaN,
                       es_deg=NaN, S_edge_L=NaN, S_edge_R=NaN,
                       string_order_1=NaN, string_order_2=NaN,
                       phase=:error))
        end
    end

    csvpath = joinpath(outdir, "phase_scan.csv")
    CSV.write(csvpath, df)
    @info "Wrote $csvpath"
    display(df)
    return df
end

# ============================================================================
# * Excited-state DMRG + per-state quantum-number diagnostics
# ----------------------------------------------------------------------------
# Used by the L-convergence study: we want E_0..E_4 plus a sector label
# (matter parity) and two operator expectations on each eigenstate.
# ============================================================================

"""
    solve_z2_higgs_k(H, sites; k_max=4, nsweeps=80, weight=30) -> (energies, psis)

DMRG for E_0..E_{k_max} via the orthogonality-weight method. Each excited level
is optimized while penalizing overlap with all previously found states.

Returns a Vector{Float64} of length k_max+1 and a Vector{MPS} of the same
length. Bond dimensions, weights, and sweep counts are conservative defaults
that work for L<=12 on the Z₂-LGT Hamiltonian we use.
"""
# Chunked DMRG with early-stopping based on |ΔE| between chunks.  After A/B
# benchmarks (test_observer_vs_chunked.jl) we found per-sweep cost is ~1.8×
# cheaper in the chunked form than in a single dmrg() call with a custom
# observer, more than offsetting the slightly higher sweep count needed to
# reach |ΔE|<tol.
function _dmrg_maybe_early(H, prev_psis, psi_init;
                           nsweeps::Int, maxdim, cutoff, noise,
                           weight::Real, early_stop_tol::Float64,
                           minsweeps::Int=20,
                           chunk_sweeps::Int=10)
    if early_stop_tol <= 0
        return weight > 0 ?
            dmrg(H, prev_psis, psi_init; nsweeps, maxdim, cutoff, noise, weight) :
            dmrg(H, psi_init; nsweeps, maxdim, cutoff, noise)
    end
    psi = psi_init
    E   = Inf
    swept = 0
    while swept < nsweeps
        nsw_this = min(chunk_sweeps, nsweeps - swept)
        E_new, psi = weight > 0 ?
            dmrg(H, prev_psis, psi;
                 nsweeps=nsw_this, maxdim, cutoff, noise, weight, outputlevel=0) :
            dmrg(H, psi; nsweeps=nsw_this, maxdim, cutoff, noise, outputlevel=0)
        swept += nsw_this
        dE = abs(E_new - E)
        E = E_new
        if swept >= minsweeps && isfinite(dE) && dE < early_stop_tol
            break
        end
    end
    return E, psi
end

function solve_z2_higgs_k(H, sites; k_max::Int=4,
                          nsweeps::Int=80,
                          weight::Real=30.0,
                          maxdim_final::Int=800,
                          cutoff::Float64=1.0e-10,
                          warm_start::Bool=false,
                          early_stop_tol::Float64=0.0,
                          physical_gap_threshold::Float64=0.0,
                          max_attempts::Int=0)
    @assert k_max >= 0
    # Ramp bond dim through the sweeps. With lambda_gauss=20 the spectrum near
    # E_0 is dense, so we give higher levels more bond and more sweeps.
    # The ramp ends at maxdim_final; any sweeps beyond stay at that value.
    maxdim = [10, 20, 40, 80, 100, 200, 300, 400, 600, min(800, maxdim_final),
              min(1000, maxdim_final), min(1200, maxdim_final),
              min(1600, maxdim_final), min(2000, maxdim_final)]
    noise  = [1e-7, 1e-8, 1e-9, 0.0]

    energies = Float64[]
    psis = MPS[]

    _t_start = time()
    psi0_init = random_mps(sites; linkdims=2)
    _t_gs = time()
    e0, psi0 = _dmrg_maybe_early(H, MPS[], psi0_init; nsweeps, maxdim, cutoff,
                                 noise, weight=0.0, early_stop_tol=early_stop_tol)
    push!(energies, e0)
    push!(psis, psi0)
    Printf.@printf("  [solve_z2] k=0/%d  E=%.5f  Δt=%.1fs  total=%.1fs\n",
                   k_max, e0, time()-_t_gs, time()-_t_start)
    flush(stdout)

    # We keep ALL found states in `psis` for orthogonality, but track which
    # are "physical" (gap to ground < physical_gap_threshold). If filtering is
    # enabled, we keep computing until we have k_max+1 physical states (or
    # hit max_attempts). Output is filtered to physical-only.
    physical_mask = Bool[true]  # ground state always physical
    target_physical = k_max + 1
    skip_enabled = physical_gap_threshold > 0
    bound = (max_attempts > 0) ? max_attempts : k_max
    attempt = 0

    while (skip_enabled ? sum(physical_mask) < target_physical : attempt < k_max) &&
          attempt < bound
        attempt += 1
        # Initial guess: random by default; warm-start uses H·ψ_{k-1}.
        psi_init = if warm_start
            try
                w = apply(H, psis[end];
                          maxdim=max(div(maxdim_final, 2), 100), cutoff=1e-8)
                normalize!(w)
                w
            catch
                random_mps(sites; linkdims=2)
            end
        else
            random_mps(sites; linkdims=2)
        end
        _t_k = time()
        ek, psik = _dmrg_maybe_early(H, psis, psi_init; nsweeps, maxdim,
                                     cutoff, noise, weight=weight,
                                     early_stop_tol=early_stop_tol)
        push!(energies, ek)
        push!(psis, psik)
        gap_k = ek - energies[1]
        is_physical = (!skip_enabled) || (gap_k <= physical_gap_threshold)
        push!(physical_mask, is_physical)
        # Progress + ETA based on average per-state time so far.
        _elapsed = time() - _t_start
        _avg_per_state = _elapsed / (attempt + 1)  # ground + attempt-many excited
        _eta = max(0.0, _avg_per_state * (k_max - attempt))
        Printf.@printf("  [solve_z2] k=%d/%d  E=%.5f  gap=%.5f  Δt=%.1fs  total=%.1fs  ETA=%.1fs%s\n",
                       attempt, k_max, ek, gap_k, time()-_t_k, _elapsed, _eta,
                       is_physical ? "" : "  [unphysical, skip]")
        flush(stdout)
    end

    # If skip filtering is on, return only the physical states.
    if skip_enabled
        idx = findall(physical_mask)
        if length(idx) < target_physical
            @warn "solve_z2_higgs_k: only found $(length(idx)) physical states out of target $(target_physical) after $(attempt) attempts"
        end
        return energies[idx], psis[idx]
    end

    return energies, psis
end

"""
    matter_link_indices(L, boundary) -> (matter_idx, link_idx)

1-based qubit indices for matter sites and links given the boundary mode.
"""
function matter_link_indices(L::Int, boundary::Symbol)
    N = num_qubits(L, boundary)
    if boundary == :open_site
        # matter: 1,3,...,2L-1 ; links: 2,4,...,2L-2
        matter_idx = collect(1:2:N)
        link_idx   = collect(2:2:(N-1))
    elseif boundary == :open_link
        # links: 1,3,...,2L+1 ; matter: 2,4,...,2L
        link_idx   = collect(1:2:N)
        matter_idx = collect(2:2:(N-1))
    elseif boundary == :PBC
        # links: 1,3,...,2L-1 ; matter: 2,4,...,2L
        link_idx   = collect(1:2:N)
        matter_idx = collect(2:2:N)
    else
        error("Unknown boundary $boundary")
    end
    return matter_idx, link_idx
end

"""
    matter_parity_value(psi, sites, boundary) -> Float64

Compute ⟨ψ| ∏_{n ∈ matter} Z_n |ψ⟩. The product runs over matter qubits only
(link qubits are skipped). The Hamiltonian commutes with this operator so the
result should be ±1 up to DMRG noise.
"""
function matter_parity_value(psi::MPS, sites, boundary::Symbol, L::Int)
    matter_idx, _ = matter_link_indices(L, boundary)
    isempty(matter_idx) && return NaN
    os = OpSum()
    ops_tuple = Any[]
    for n in matter_idx
        push!(ops_tuple, "Z"); push!(ops_tuple, n)
    end
    os += (1.0, ops_tuple...)
    P = MPO(os, sites)
    return real(inner(psi', P, psi))
end

"""
    sum_local_op(psi, op, indices) -> Float64

Sum of single-site expectation values of `op` over the given qubit indices.
Uses `expect(psi, op)` which returns one number per site of the MPS.
"""
function sum_local_op(psi::MPS, op::String, indices::AbstractVector{<:Integer})
    vals = expect(psi, op)
    return sum(real(vals[i]) for i in indices)
end

"""
    state_diagnostics(psi, sites, boundary, L) -> NamedTuple

Per-state quantum-number diagnostics:
  matter_parity   = <ψ| ∏_n Z_n |ψ>        (n over matter sites)
  total_Z_matter  = <ψ| Σ_n Z_n |ψ>
  total_X_link    = <ψ| Σ_l X_l |ψ>        (l over links)
"""
function state_diagnostics(psi::MPS, sites, boundary::Symbol, L::Int)
    matter_idx, link_idx = matter_link_indices(L, boundary)
    P_match = matter_parity_value(psi, sites, boundary, L)
    Zsum = isempty(matter_idx) ? NaN : sum_local_op(psi, "Z", matter_idx)
    Xsum = isempty(link_idx)   ? NaN : sum_local_op(psi, "X", link_idx)
    return (matter_parity=P_match, total_Z_matter=Zsum, total_X_link=Xsum)
end

# ============================================================================
# * convergence_run: L-convergence of low-lying spectrum + quantum-number labels
# ============================================================================

"""
    convergence_run(; Larr=[4,6,8,10,12], k_max=4,
                    z2_obc_boundary=:drop, outdir=...) -> DataFrame

For each L and each boundary ∈ {:open_site, :PBC} compute E_0..E_{k_max} with
the Eq.2 Z₂-LGT Hamiltonian at the paper reference point
(m_0 = 1.125, η = 1, α = 1). For each eigenstate compute the three
diagnostics from `state_diagnostics`. Writes one CSV per call.

The `z2_obc_boundary` knob only modifies the OBC Hamiltonian.
"""
function convergence_run(; Larr=[4, 6, 8, 10, 12],
                         k_max::Int=4,
                         z2_obc_boundary::Symbol=:drop,
                         outdir=joinpath(@__DIR__, "data", "z2_gauge_theory"))
    mkpath(outdir)
    @info "convergence_run" Larr k_max z2_obc_boundary
    df = DataFrame(L=Int[], boundary=Symbol[], z2_obc_boundary=Symbol[],
                   k=Int[], E_k=Float64[], gap_k=Float64[],
                   matter_parity=Float64[],
                   total_Z_matter=Float64[],
                   total_X_link=Float64[])

    for L in Larr, boundary in (:open_site, :PBC)
        # Reference point: m_0=1.125, eta=1, alpha=1.
        p = Data.Params(; m0=1.125, eta=1.0, alpha=1.0, Lphys=L, L=L)
        @info "convergence_run: L=$L, boundary=$boundary"
        try
            H, sites = build_H(p; boundary=boundary, gauge_law=:z2,
                               z2_obc_boundary=z2_obc_boundary)
            energies, psis = solve_z2_higgs_k(H, sites; k_max=k_max)
            E0 = energies[1]
            for k in 0:k_max
                diag = state_diagnostics(psis[k+1], sites, boundary, L)
                push!(df, (L=L, boundary=boundary,
                           z2_obc_boundary=z2_obc_boundary,
                           k=k, E_k=energies[k+1],
                           gap_k=energies[k+1] - E0,
                           matter_parity=diag.matter_parity,
                           total_Z_matter=diag.total_Z_matter,
                           total_X_link=diag.total_X_link))
            end
        catch err
            @warn "convergence_run point failed" L boundary exception=(err, catch_backtrace())
            for k in 0:k_max
                push!(df, (L=L, boundary=boundary,
                           z2_obc_boundary=z2_obc_boundary,
                           k=k, E_k=NaN, gap_k=NaN,
                           matter_parity=NaN,
                           total_Z_matter=NaN,
                           total_X_link=NaN))
            end
        end
    end

    csvpath = joinpath(outdir, "convergence_$(z2_obc_boundary).csv")
    CSV.write(csvpath, df)
    @info "Wrote $csvpath"
    display(df)
    return df
end

# ============================================================================
# * Meson-sector gap via two methods
# ----------------------------------------------------------------------------
# Section III mass M = E_P - E_Omega (paper Eq. 27): the "meson" is the lowest
# excited state in the SAME (matter_parity, total_Z_matter) sector as the
# ground state. We provide two complementary methods:
#
#   Option 1: sector_filtered — find lowest k>=1 (out of k_max excited states)
#             whose (matter_parity, total_Z_matter) matches the ground state.
#   Option 2: overlap         — build O_center = X_m * sigma^z_{m,m+1} * X_{m+1}
#             at the chain centre, apply to ground state |Omega> to get |P>,
#             then find the eigenstate ψ_k with the largest |<ψ_k|P>|^2.
#
# Both methods share the same heavy DMRG step (k_max excited states). To avoid
# re-running DMRG twice per (L,boundary,z2_obc) we compute (energies, psis)
# once and pass them to both reporters.
# ============================================================================

"""
    center_3site_indices(L, boundary) -> (a, b, c)

Return three consecutive qubit indices (matter, link, matter) at the centre of
the chain. Used for building O_center = X_a * Z_b * X_c.

For :open_site, matter occupies odd qubit indices (1,3,...,2L-1); the middle
matter pair is matter[m], matter[m+1] with m = L÷2. Their qubit indices are
2m-1 and 2m+1, and the link between them is at 2m.

For :PBC and :open_link, matter occupies even indices (2,4,...). The middle
matter pair is at qubits 2m and 2m+2, link between them at 2m+1.

Cross-check caveat: m = L÷2 (integer division). For even L the operator sits
slightly left of dead-centre; this is the cleanest deterministic convention.
"""
function center_3site_indices(L::Int, boundary::Symbol)
    @assert L >= 2 "Need L >= 2 to place a 3-site operator"
    m = L ÷ 2
    if boundary == :open_site
        a = 2m - 1
        b = 2m
        c = 2m + 1
    else  # :PBC or :open_link, matter at even indices
        a = 2m
        b = 2m + 1
        c = 2m + 2
    end
    return (a, b, c)
end

"""
    make_O_center_mpo(sites, L, boundary) -> MPO

Build MPO for O_center = X * Z * X on the central matter-link-matter triple.
"""
function make_O_center_mpo(sites, L::Int, boundary::Symbol)
    a, b, c = center_3site_indices(L, boundary)
    N = length(sites)
    @assert 1 <= a && c <= N "O_center indices out of range: ($a,$b,$c) vs N=$N"
    os = OpSum()
    os += 1.0, "X", a, "Z", b, "X", c
    return MPO(os, sites)
end

"""
    make_O_pzero_mpo(sites, L, boundary) -> MPO

Momentum-zero meson creation operator:
  O_p=0 = (1/√n_terms) Σ_j X_{matter j}·σᶻ_{j,j+1}·X_{matter j+1}
:open_site → matter at odd qubits 2j-1, link at 2j; n_terms = L-1.
:PBC       → matter at even qubits 2j, link at 2j+1 (odd); n_terms = L,
             with wrap-around term at (2L, 1, 2).
"""
function make_O_pzero_mpo(sites, L::Int, boundary::Symbol)
    n_terms = boundary == :PBC ? L : (L - 1)
    nrm = 1.0 / sqrt(n_terms)
    os = OpSum()
    if boundary == :open_site
        for j in 1:(L-1)
            os += nrm, "X", 2j-1, "Z", 2j, "X", 2j+1
        end
    elseif boundary == :PBC
        for j in 1:(L-1)
            os += nrm, "X", 2j, "Z", 2j+1, "X", 2j+2
        end
        # wrap-around: matter j=L at 2L, closing link at 1, matter j=1 at 2
        os += nrm, "X", 2L, "Z", 1, "X", 2
    else
        error("make_O_pzero_mpo: only :open_site and :PBC supported (got $boundary)")
    end
    return MPO(os, sites)
end

"""
    compute_sector_meson(energies, psis, diagnostics; parity_tol=0.05, z_tol=0.5)
        -> (k_meson, gap_meson, P_meson, totalZ_meson, totalX_meson)

Given an energy list, MPS list, and a per-state diagnostics list (NamedTuples
from `state_diagnostics`), find the lowest k>=1 such that
  - |matter_parity(k) - matter_parity(0)| <= parity_tol
  - |total_Z_matter(k)| <= z_tol
Returns (-1, NaN, NaN, NaN, NaN) if none of the available k's satisfy both.
"""
function compute_sector_meson(energies, psis, diagnostics;
                              parity_tol::Real=0.05,
                              z_tol::Real=0.5)
    @assert length(energies) == length(diagnostics)
    P0 = diagnostics[1].matter_parity
    for k in 2:length(energies)
        Pk = diagnostics[k].matter_parity
        Zk = diagnostics[k].total_Z_matter
        if abs(Pk - P0) <= parity_tol && abs(Zk) <= z_tol
            return (k - 1, energies[k] - energies[1],
                    Pk, Zk, diagnostics[k].total_X_link)
        end
    end
    return (-1, NaN, NaN, NaN, NaN)
end

# ============================================================================
# * convergence_sector_filtered (Option 1): higher-k sector-filtered gap
# ============================================================================

"""
    convergence_sector_filtered(; Larr, k_max, boundaries, z2_obc_boundaries,
                                outdir) -> DataFrame

For each (L, boundary, z2_obc_boundary), compute E_0..E_{k_max} via DMRG,
diagnose (matter_parity, total_Z_matter, total_X_link) on each state, and
report the lowest k>=1 in the same charge-neutral sector as the ground state.

Writes data/z2_gauge_theory/convergence_sector.csv.
"""
function convergence_sector_filtered(; Larr=[4, 6, 8, 10, 12],
                                     k_max::Int=12,
                                     boundaries=(:open_site, :PBC),
                                     z2_obc_boundaries=(:drop, :truncate_xz),
                                     parity_tol::Real=0.05,
                                     z_tol::Real=0.5,
                                     nsweeps::Int=16,
                                     weight::Real=50.0,
                                     outdir=joinpath(@__DIR__, "data", "z2_gauge_theory"))
    mkpath(outdir)
    @info "convergence_sector_filtered" Larr k_max boundaries z2_obc_boundaries
    df = DataFrame(L=Int[], boundary=Symbol[], z2_obc_boundary=Symbol[],
                   k_meson=Int[], gap_meson=Float64[],
                   P_meson=Float64[], totalZ_meson=Float64[],
                   totalX_meson=Float64[], k_max_searched=Int[],
                   converged_note=String[])

    for L in Larr, boundary in boundaries, z2obc in z2_obc_boundaries
        # PBC: z2_obc_boundary is irrelevant. Only run once per L for PBC
        # (with z2obc = :drop as a placeholder).
        if boundary == :PBC && z2obc != :drop
            continue
        end

        p = Data.Params(; m0=1.125, eta=1.0, alpha=1.0, Lphys=L, L=L)
        @info "sector_filtered: L=$L, boundary=$boundary, z2obc=$z2obc"
        local energies, psis, diagnostics
        note = ""
        try
            H, sites = build_H(p; boundary=boundary, gauge_law=:z2,
                               z2_obc_boundary=z2obc)
            energies, psis = solve_z2_higgs_k(H, sites; k_max=k_max,
                                              nsweeps=nsweeps, weight=weight)
            diagnostics = [state_diagnostics(psis[k+1], sites, boundary, L)
                           for k in 0:k_max]
            (km, gm, Pm, Zm, Xm) = compute_sector_meson(energies, psis,
                                                       diagnostics;
                                                       parity_tol=parity_tol,
                                                       z_tol=z_tol)
            # Convergence sanity: variance / residual for highest state.
            # We use ||H psi - E psi||^2 on the top excited state as a proxy.
            try
                ek_top = energies[end]
                # ||Hpsi - E*psi||^2 = <Hpsi|Hpsi> - 2 E <psi|Hpsi> + E^2
                # Use inner(H', ψ, H, ψ) / inner(ψ', H, ψ) to skip the
                # explicit apply(H, ψ).
                hh = real(inner(H, psis[end], H, psis[end]))
                he = real(inner(psis[end]', H, psis[end]))
                var_top = hh - 2 * ek_top * he + ek_top^2
                if var_top > 1e-3
                    note *= "top-state res=$(round(var_top; sigdigits=3)); "
                end
            catch err
                note *= "var-check failed ($(typeof(err))); "
            end
            push!(df, (L=L, boundary=boundary, z2_obc_boundary=z2obc,
                       k_meson=km, gap_meson=gm,
                       P_meson=Pm, totalZ_meson=Zm, totalX_meson=Xm,
                       k_max_searched=k_max, converged_note=note))
        catch err
            @warn "sector_filtered failed" L boundary z2obc exception=(err, catch_backtrace())
            push!(df, (L=L, boundary=boundary, z2_obc_boundary=z2obc,
                       k_meson=-1, gap_meson=NaN,
                       P_meson=NaN, totalZ_meson=NaN, totalX_meson=NaN,
                       k_max_searched=k_max,
                       converged_note="ERROR: $(sprint(showerror, err))"))
        end
    end

    csvpath = joinpath(outdir, "convergence_sector.csv")
    CSV.write(csvpath, df)
    @info "Wrote $csvpath"
    display(df)
    return df
end

# ============================================================================
# * convergence_overlap (Option 2): O_center |Omega> overlap targeting
# ============================================================================

"""
    compute_overlap_meson(energies, psis, sites, L, boundary)
        -> (k_dominant, gap_dominant, overlap_dominant, overlaps::Vector{Float64})

Build O_center MPO, compute |P> = O_center|psi_0> (as MPS via apply, normalized),
and return overlaps |<psi_k|P>|^2 = |<psi_k|O_center|psi_0>|^2 / <P|P>
for k = 0..K where K = length(psis)-1. Returns the k>=0 with the largest
overlap (ground state can dominate if O is trivial on |Omega>).

Cross-check caveat: we DO renormalize |P>, so overlaps sum to <=1 (they should
sum to 1 if the K+1 states span the relevant Krylov subspace). The "dominant k"
ranking is the same regardless of normalization. Operator ordering inside the
MPO is X_a Z_b X_c (lexicographic in qubit index), which equals the paper
ordering X_m sigma^z_{m,m+1} X_{m+1} for our centre-pair convention.
"""
function compute_overlap_meson(energies, psis, sites, L::Int, boundary::Symbol)
    O = make_O_center_mpo(sites, L, boundary)
    psi0 = psis[1]
    P = apply(O, psi0)
    nrm2 = real(inner(P, P))
    if !(nrm2 > 1e-14)
        return (-1, NaN, NaN, fill(NaN, length(psis)))
    end
    Pn = P / sqrt(nrm2)
    overlaps = Float64[]
    for psik in psis
        ov = inner(psik, Pn)
        push!(overlaps, abs(ov)^2)
    end
    k_dom = argmax(overlaps) - 1
    gap_dom = energies[k_dom + 1] - energies[1]
    return (k_dom, gap_dom, overlaps[k_dom + 1], overlaps)
end

"""
    convergence_overlap(; Larr, k_max, boundaries, z2_obc_boundaries, outdir)
        -> DataFrame

Same parameter grid as convergence_sector_filtered. Runs DMRG, builds the
3-site operator O_center at the chain centre, applies to the ground state,
and reports the eigenstate with maximum overlap.

Writes data/z2_gauge_theory/convergence_overlap.csv.
"""
function convergence_overlap(; Larr=[4, 6, 8, 10, 12],
                             k_max::Int=12,
                             boundaries=(:open_site, :PBC),
                             z2_obc_boundaries=(:drop, :truncate_xz),
                             nsweeps::Int=16,
                             weight::Real=50.0,
                             outdir=joinpath(@__DIR__, "data", "z2_gauge_theory"))
    mkpath(outdir)
    @info "convergence_overlap" Larr k_max boundaries z2_obc_boundaries
    # CSV with k_max+1 overlap columns. We pre-fix to k_max so all rows align.
    overlap_cols = [Symbol("overlap_k$(k)") for k in 0:k_max]
    col_pairs = vcat([:L => Int[], :boundary => Symbol[],
                      :z2_obc_boundary => Symbol[],
                      :k_dominant => Int[], :gap_dominant => Float64[],
                      :overlap_dominant => Float64[]],
                     [c => Float64[] for c in overlap_cols],
                     [:converged_note => String[]])
    df = DataFrame(; (Symbol(p.first) => p.second for p in col_pairs)...)

    for L in Larr, boundary in boundaries, z2obc in z2_obc_boundaries
        if boundary == :PBC && z2obc != :drop
            continue
        end
        p = Data.Params(; m0=1.125, eta=1.0, alpha=1.0, Lphys=L, L=L)
        @info "overlap: L=$L, boundary=$boundary, z2obc=$z2obc"
        note = ""
        try
            H, sites = build_H(p; boundary=boundary, gauge_law=:z2,
                               z2_obc_boundary=z2obc)
            energies, psis = solve_z2_higgs_k(H, sites; k_max=k_max,
                                              nsweeps=nsweeps, weight=weight)
            (kd, gapd, ovd, overlaps) = compute_overlap_meson(energies, psis,
                                                              sites, L, boundary)
            row = Dict{Symbol,Any}(:L => L, :boundary => boundary,
                                   :z2_obc_boundary => z2obc,
                                   :k_dominant => kd,
                                   :gap_dominant => gapd,
                                   :overlap_dominant => ovd,
                                   :converged_note => note)
            for k in 0:k_max
                row[Symbol("overlap_k$(k)")] = k+1 <= length(overlaps) ? overlaps[k+1] : NaN
            end
            push!(df, row)
        catch err
            @warn "overlap failed" L boundary z2obc exception=(err, catch_backtrace())
            row = Dict{Symbol,Any}(:L => L, :boundary => boundary,
                                   :z2_obc_boundary => z2obc,
                                   :k_dominant => -1,
                                   :gap_dominant => NaN,
                                   :overlap_dominant => NaN,
                                   :converged_note => "ERROR: $(sprint(showerror, err))")
            for k in 0:k_max
                row[Symbol("overlap_k$(k)")] = NaN
            end
            push!(df, row)
        end
    end

    csvpath = joinpath(outdir, "convergence_overlap.csv")
    CSV.write(csvpath, df)
    @info "Wrote $csvpath"
    display(df)
    return df
end

# ============================================================================
# * convergence_overlap_pzero: momentum-zero version of Option 2
# ============================================================================

"""
    compute_overlap_meson_pzero(energies, psis, sites, L, boundary;
                                 overlap_threshold=0.01)
        -> (k_meson, gap_meson, ov_meson, k_argmax, gap_argmax, ov_argmax,
            P_norm, overlaps::Vector{Float64})

Build O_p=0 MPO, compute |P_p=0⟩ = O_p=0|ψ_0⟩, return per-state overlaps
|⟨ψ_k|O_p=0|ψ_0⟩|² (unnormalized — O_p=0 is NOT unitary so |P⟩ norm may differ
from 1; we also report it). Two pick rules:
  - k_meson: lowest k ≥ 1 with overlap ≥ threshold (the physically motivated
             choice — gives the lowest-energy state that the operator actually
             creates with non-negligible weight).
  - k_argmax: argmax over k ≥ 1 (for comparison with the localized O variant).
"""
function compute_overlap_meson_pzero(energies, psis, sites, L::Int, boundary::Symbol;
                                     overlap_threshold::Real=0.01)
    O = make_O_pzero_mpo(sites, L, boundary)
    psi0 = psis[1]
    P = apply(O, psi0)
    Pnorm = sqrt(max(real(inner(P, P)), 0.0))
    overlaps = Float64[]
    for psik in psis
        amp = inner(psik, P)        # = ⟨ψ_k | O | ψ_0⟩, since |P⟩ = O|ψ_0⟩
        push!(overlaps, abs(amp)^2)
    end
    k_meson = -1; gap_meson = NaN; ov_meson = NaN
    for k in 1:(length(psis)-1)
        if isfinite(overlaps[k+1]) && overlaps[k+1] >= overlap_threshold
            k_meson   = k
            gap_meson = energies[k+1] - energies[1]
            ov_meson  = overlaps[k+1]
            break
        end
    end
    k_argmax = -1; max_ov = -Inf
    for k in 1:(length(psis)-1)
        if isfinite(overlaps[k+1]) && overlaps[k+1] > max_ov
            max_ov   = overlaps[k+1]
            k_argmax = k
        end
    end
    gap_argmax = (k_argmax == -1) ? NaN : energies[k_argmax+1] - energies[1]
    ov_argmax  = (k_argmax == -1) ? NaN : overlaps[k_argmax+1]
    return (k_meson, gap_meson, ov_meson, k_argmax, gap_argmax, ov_argmax,
            Pnorm, overlaps)
end

function convergence_overlap_pzero(; Larr=[4, 6, 8, 10, 12],
                                   k_max::Int=12,
                                   boundaries=(:open_site, :PBC),
                                   z2_obc_boundaries=(:drop, :truncate_xz),
                                   nsweeps::Int=80,
                                   weight::Real=200.0,
                                   maxdim_final::Int=300,
                                   dmrg_cutoff::Float64=1e-11,
                                   overlap_threshold::Float64=0.01,
                                   bg_left::Int=+1,
                                   bg_right::Int=+1,
                                   m0::Float64=1.125,
                                   eta::Float64=1.0,
                                   alpha::Float64=1.0,
                                   seed::Int=0,
                                   warm_start::Bool=false,
                                   early_stop_tol::Float64=1e-7,
                                   physical_gap_threshold::Float64=0.0,
                                   max_attempts::Int=0,
                                   fname_suffix::String="",
                                   outdir=joinpath(@__DIR__, "data", "z2_gauge_theory"))
    mkpath(outdir)
    if seed > 0
        Random.seed!(seed)
        @info "convergence_overlap_pzero seeded" seed
    end
    @info "convergence_overlap_pzero" Larr k_max nsweeps weight overlap_threshold bg_left bg_right
    overlap_cols = [Symbol("overlap_k$(k)") for k in 0:k_max]
    sigma_cols   = [Symbol("sigma_k$(k)")   for k in 0:k_max]
    gap_cols     = [Symbol("gap_k$(k)")     for k in 0:k_max]
    col_pairs = vcat([:L => Int[], :boundary => Symbol[],
                      :z2_obc_boundary => Symbol[],
                      :k_meson => Int[], :gap_meson => Float64[],
                      :overlap_meson => Float64[], :sigma_meson => Float64[],
                      :k_argmax => Int[], :gap_argmax => Float64[],
                      :overlap_argmax => Float64[], :sigma_argmax => Float64[],
                      :barE_gap => Float64[], :sigma_barE => Float64[],
                      :W_total => Float64[],
                      :P_norm => Float64[]],
                     [c => Float64[] for c in overlap_cols],
                     [c => Float64[] for c in sigma_cols],
                     [c => Float64[] for c in gap_cols],
                     [:converged_note => String[]])
    df = DataFrame(; (Symbol(p.first) => p.second for p in col_pairs)...)

    for L in Larr, boundary in boundaries, z2obc in z2_obc_boundaries
        if boundary == :PBC && z2obc != :drop
            continue
        end
        p = Data.Params(; m0=m0, eta=eta, alpha=alpha, Lphys=L, L=L)
        @info "overlap_pzero: L=$L, boundary=$boundary, z2obc=$z2obc, m0=$m0, eta=$eta"
        try
            _t_h = @elapsed (H, sites) = build_H(p; boundary=boundary, gauge_law=:z2,
                               z2_obc_boundary=z2obc,
                               bg_left=bg_left, bg_right=bg_right)
            _t_dmrg = @elapsed (energies, psis) = solve_z2_higgs_k(H, sites; k_max=k_max,
                                              nsweeps=nsweeps, weight=weight,
                                              maxdim_final=maxdim_final,
                                              cutoff=dmrg_cutoff,
                                              warm_start=warm_start,
                                              early_stop_tol=early_stop_tol,
                                              physical_gap_threshold=physical_gap_threshold,
                                              max_attempts=max_attempts)
            _t_ov = @elapsed (km, gm, om, ka, ga, oa, pn, overlaps) =
                compute_overlap_meson_pzero(energies, psis, sites, L, boundary;
                                            overlap_threshold=overlap_threshold)
            # Per-state energy variance σ_k = √(⟨ψ_k|H²|ψ_k⟩ − E_k²).
            _t_var = @elapsed begin
                sigmas = Float64[]
                for (k, ψk) in enumerate(psis)
                    try
                        h2 = real(inner(H, ψk, H, ψk))
                        σ2 = h2 - energies[k]^2
                        push!(sigmas, sqrt(max(σ2, 0.0)))
                    catch
                        push!(sigmas, NaN)
                    end
                end
            end
            Printf.@printf("  [PROFILE] L=%d %s %s: H_build=%.2fs  DMRG=%.2fs  overlap=%.2fs  variance=%.2fs\n",
                           L, string(boundary), string(z2obc),
                           _t_h, _t_dmrg, _t_ov, _t_var)
            flush(stdout)
            σ_gs = sigmas[1]
            σ_meson  = km == -1 ? NaN : sqrt(sigmas[km + 1]^2 + σ_gs^2)
            σ_argmax = ka == -1 ? NaN : sqrt(sigmas[ka + 1]^2 + σ_gs^2)
            # Per-k gap_k = E_k - E_0.
            E0 = energies[1]
            gaps_per_k = Float64[energies[k+1] - E0 for k in 0:k_max]
            # Spectral first moment ̄gap = Σ_{k≥1} w_k·gap_k / Σ_{k≥1} w_k.
            sum_w = 0.0; sum_Ew = 0.0; sum_σ²w² = 0.0
            for k in 1:k_max
                kidx = k + 1
                if kidx <= length(overlaps) && isfinite(overlaps[kidx])
                    w = overlaps[kidx]
                    sum_w  += w
                    sum_Ew += gaps_per_k[kidx] * w
                    σk² = (kidx <= length(sigmas) && isfinite(sigmas[kidx])) ? sigmas[kidx]^2 : 0.0
                    sum_σ²w² += σk² * w^2
                end
            end
            barE_gap_  = sum_w > 0 ? sum_Ew / sum_w : NaN
            σ_barE_    = sum_w > 0 ? sqrt(sum_σ²w² / sum_w^2 + σ_gs^2) : NaN
            W_total_   = sum_w
            row = Dict{Symbol,Any}(:L => L, :boundary => boundary,
                                   :z2_obc_boundary => z2obc,
                                   :k_meson => km, :gap_meson => gm,
                                   :overlap_meson => om, :sigma_meson => σ_meson,
                                   :k_argmax => ka, :gap_argmax => ga,
                                   :overlap_argmax => oa, :sigma_argmax => σ_argmax,
                                   :barE_gap => barE_gap_, :sigma_barE => σ_barE_,
                                   :W_total => W_total_,
                                   :P_norm => pn,
                                   :converged_note => "")
            for k in 0:k_max
                row[Symbol("overlap_k$(k)")] = k+1 <= length(overlaps) ? overlaps[k+1] : NaN
                row[Symbol("sigma_k$(k)")]   = k+1 <= length(sigmas)   ? sigmas[k+1]   : NaN
                row[Symbol("gap_k$(k)")]     = k+1 <= length(gaps_per_k) ? gaps_per_k[k+1] : NaN
            end
            push!(df, row)
        catch err
            @warn "overlap_pzero failed" L boundary z2obc exception=(err, catch_backtrace())
            row = Dict{Symbol,Any}(:L => L, :boundary => boundary,
                                   :z2_obc_boundary => z2obc,
                                   :k_meson => -1, :gap_meson => NaN,
                                   :overlap_meson => NaN, :sigma_meson => NaN,
                                   :k_argmax => -1, :gap_argmax => NaN,
                                   :overlap_argmax => NaN, :sigma_argmax => NaN,
                                   :barE_gap => NaN, :sigma_barE => NaN,
                                   :W_total => NaN,
                                   :P_norm => NaN,
                                   :converged_note => "ERROR: $(sprint(showerror, err))")
            for k in 0:k_max
                row[Symbol("overlap_k$(k)")] = NaN
                row[Symbol("sigma_k$(k)")]   = NaN
                row[Symbol("gap_k$(k)")]     = NaN
            end
            push!(df, row)
        end
    end

    csvpath = joinpath(outdir, "convergence_overlap_pzero$(fname_suffix).csv")
    CSV.write(csvpath, df)
    @info "Wrote $csvpath"
    display(df)
    return df
end

# ============================================================================
# * bg_charge_scan_pzero: scan (bg_left, bg_right) ∈ {±1}² at fixed L,
#   OBC :truncate_xz, with momentum-zero overlap targeting.
# ============================================================================

function bg_charge_scan_pzero(; L::Int=8, k_max::Int=12,
                                nsweeps::Int=28, weight::Real=100.0,
                                overlap_threshold::Float64=0.01,
                                outdir=joinpath(@__DIR__, "data", "z2_gauge_theory"))
    mkpath(outdir)
    @info "bg_charge_scan_pzero" L k_max
    overlap_cols = [Symbol("overlap_k$(k)") for k in 0:k_max]
    col_pairs = vcat([:L => Int[], :bg_left => Int[], :bg_right => Int[],
                      :E0 => Float64[], :gap1 => Float64[], :gap2 => Float64[],
                      :k_meson => Int[], :gap_meson => Float64[], :overlap_meson => Float64[],
                      :k_argmax => Int[], :gap_argmax => Float64[], :overlap_argmax => Float64[],
                      :P_norm => Float64[]],
                     [c => Float64[] for c in overlap_cols])
    df = DataFrame(; (Symbol(p.first) => p.second for p in col_pairs)...)

    for bg_left in (+1, -1), bg_right in (+1, -1)
        @info "bg_scan: L=$L  bg_left=$bg_left  bg_right=$bg_right"
        p = Data.Params(; m0=1.125, eta=1.0, alpha=1.0, Lphys=L, L=L)
        H, sites = build_H(p; boundary=:open_site, gauge_law=:z2,
                           z2_obc_boundary=:truncate_xz,
                           bg_left=bg_left, bg_right=bg_right)
        energies, psis = solve_z2_higgs_k(H, sites; k_max=k_max,
                                          nsweeps=nsweeps, weight=weight)
        E0   = energies[1]
        gap1 = energies[2] - E0
        gap2 = energies[3] - E0
        (km, gm, om, ka, ga, oa, pn, overlaps) =
            compute_overlap_meson_pzero(energies, psis, sites, L, :open_site;
                                        overlap_threshold=overlap_threshold)
        row = Dict{Symbol,Any}(:L => L, :bg_left => bg_left, :bg_right => bg_right,
                               :E0 => E0, :gap1 => gap1, :gap2 => gap2,
                               :k_meson => km, :gap_meson => gm, :overlap_meson => om,
                               :k_argmax => ka, :gap_argmax => ga, :overlap_argmax => oa,
                               :P_norm => pn)
        for k in 0:k_max
            row[Symbol("overlap_k$(k)")] = k+1 <= length(overlaps) ? overlaps[k+1] : NaN
        end
        push!(df, row)
        @info "result" bg_left bg_right gap1 gap2 gap_meson=gm overlap_meson=om gap_argmax=ga overlap_argmax=oa
    end

    csvpath = joinpath(outdir, "bg_charge_scan_L$(L).csv")
    CSV.write(csvpath, df)
    @info "Wrote $csvpath"
    display(df)
    return df
end

# ============================================================================
# * main: only runs when this file is executed as a script
# ============================================================================
function main()
    # Pre-existing drivers (CSVs from previous runs are kept on disk; re-enable
    # if you want a full re-baseline). They are gated to skip by default
    # because the new sector/overlap drivers at k_max=12 are the focus.
    if get(ENV, "Z2GT_RUN_LEGACY", "false") == "true"
        @info "Starting section_iii_smoke_run (higgs_obc_boundary=:drop)"
        section_iii_smoke_run()
        @info "Starting section_iii_smoke_run (higgs_obc_boundary=:truncate_xx)"
        section_iii_smoke_run(higgs_obc_boundary=:truncate_xx)
        @info "Starting phase_scan_smoke"
        phase_scan_smoke()
        @info "Starting L-convergence run (z2_obc_boundary=:drop)"
        convergence_run(; z2_obc_boundary=:drop)
        @info "Starting L-convergence run (z2_obc_boundary=:truncate_xz)"
        convergence_run(; z2_obc_boundary=:truncate_xz)
    end
    @info "Starting convergence_sector_filtered (Option 1)"
    df_sector = convergence_sector_filtered()
    @info "Starting convergence_overlap (Option 2)"
    df_overlap = convergence_overlap()
    @info "Starting convergence_overlap_pzero (momentum-zero meson)"
    df_overlap_pzero = convergence_overlap_pzero()
    return df_sector, df_overlap, df_overlap_pzero
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
