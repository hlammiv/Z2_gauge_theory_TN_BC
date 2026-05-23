###############################################################################
# SPT / Topological Phase Diagnostics via DMRG (ITensor.jl)
#
# Two gauge-law conventions are implemented (selected via `gauge_law=`):
#
#   :z2 (paper Eq. 2 — default)
#       H = (1/2) Σ σˣ_{n,n+1}
#         - (m_0/2) Σ (-1)^n Z_n
#         + (η/4)  Σ (X_n X_{n+1} + Y_n Y_{n+1}) σᶻ_{n,n+1}
#
#   :higgs_spt   (XXX / ZZZ Z2×Z2 SPT-style cluster form)
#       H =  g · Σ X_{link} X_{site} X_{link'}
#          + h · Σ Z_{site} Z_{link} Z_{site'}
#          - α  · Σ X_{site}
#       with g = h = -1 (so both 3-body terms favor the cluster
#       ground state).  α plays the role of a transverse field
#       perturbing the SPT cluster state on the matter sublattice.
#
# Two boundary modes (selected via `boundary=`):
#
#   :open_site   (OBC, sites at the edges) — chain length N_total = 2 N_matter - 1
#                Layout:  M B M B M B ... M    (matter sites at both ends)
#   :PBC         — chain length N_total = 2 N_matter (extra closing link site)
#                Layout:  M B M B M B ... M B  (last bond site closes the ring,
#                connecting matter site N_matter back to matter site 1)
#
# For PBC we keep the MPS open with the wrap-around terms implemented as
# long-range MPO operators (ITensor handles arbitrary-range OpSum terms).
# This is not a true periodic-MPS solver but it is correct (just costlier
# in bond dimension than OBC).
#
# Diagnostics:
#   1. Entanglement spectrum (Schmidt values at each bond cut)
#   2. Entanglement entropy profile S(x)
#   3. Gap via excited-state DMRG
#   4. String order parameter
#   5. Edge entanglement (boundary site entropy)
#   6. Central charge fit at criticality
#   7. Phase classification wrapper `diagnose(...)`
#   8. Section III smoke driver (gap vs L for OBC/PBC × {z2,higgs_spt})
###############################################################################

using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using HDF5
using Random
using Statistics: mean

# ── BLAS thread setting (overridable via Z2C_BLAS_THREADS env var).
# Default: half of physical/logical threads to avoid oversubscription.
let n = parse(Int, get(ENV, "Z2C_BLAS_THREADS", string(max(1, Sys.CPU_THREADS ÷ 2))))
    BLAS.set_num_threads(n)
    @info "Z2C: BLAS threads set to $n"
end

###############################################################################
# 1. SITE INDEXING CONVENTION
###############################################################################

"""
    build_sites(N_matter; boundary=:open_site)

Return the ITensor `siteinds("Qubit", N_total)` for the alternating
matter/bond chain.

- `:open_site`  →  N_total = 2 N_matter - 1   (matter sites at both ends)
- `:PBC`        →  N_total = 2 N_matter        (extra closing link)
"""
function build_sites(N_matter::Int; boundary::Symbol=:open_site)
    if boundary === :open_site
        N_total = 2 * N_matter - 1
    elseif boundary === :PBC
        N_total = 2 * N_matter
    else
        error("Unknown boundary $boundary (use :open_site or :PBC)")
    end
    return siteinds("Qubit", N_total)
end

# Matter site n in the *physical* labelling 1..N_matter maps to chain index 2n-1
matter_idx(n::Int) = 2n - 1
# Bond site (n,n+1) maps to chain index 2n
bond_idx(n::Int)   = 2n

###############################################################################
# 2. HAMILTONIAN MPO
###############################################################################

"""
    build_hamiltonian(sites, m, η; boundary=:open_site, gauge_law=:z2, α=1.0,
                      λ_gauss=20.0, higgs_obc_boundary=:drop, z2_obc_boundary=:drop)

Build the MPO of either the paper's Z2 LGT Hamiltonian (`gauge_law=:z2`,
default) or a Z2×Z2 cluster SPT Hamiltonian (`gauge_law=:higgs_spt`).

Parameters `m` and `η` are used by `:z2`.  Parameter `α` is used by
`:higgs_spt` (the on-site transverse-field strength).  For `:higgs_spt`
the 3-body couplings are hardcoded to g = h = -1.

`z2_obc_boundary` controls the Z₂-LGT Gauss-law penalty at OBC ends:
  - `:drop`        (default): no penalty at n=1, n=N_matter (current behaviour);
                              OBC has 2 fewer Gauss-law constraints than PBC.
  - `:truncate_xz`: add partial 2-qubit Gauss-law generators at the ends,
                    treating the missing boundary link as a fixed +1 background
                    gauge field.  Restores per-site constraint count to match
                    PBC.  Only effective for `gauge_law=:z2` and OBC.
"""
function build_hamiltonian(sites, m::Float64, η::Float64;
                           boundary::Symbol=:open_site,
                           gauge_law::Symbol=:z2,
                           α::Float64=1.0,
                           λ_gauss::Float64=20.0,
                           higgs_obc_boundary::Symbol=:drop,
                           z2_obc_boundary::Symbol=:drop,
                           bg_left::Int=+1,
                           bg_right::Int=+1)
    N_total = length(sites)
    if boundary === :open_site
        @assert isodd(N_total) "open_site expects 2N_matter-1 sites"
        N_matter = (N_total + 1) ÷ 2
        n_links  = N_matter - 1
        link_n   = n -> bond_idx(n)                 # link between matter n and n+1
        # neighbour matter site (no wrap)
        next_matter = n -> (n < N_matter ? n + 1 : nothing)
    elseif boundary === :PBC
        @assert iseven(N_total) "PBC expects 2N_matter sites"
        N_matter = N_total ÷ 2
        n_links  = N_matter
        # link between matter n and n+1 lives at chain index 2n
        # the closing link n=N_matter is at chain index 2*N_matter
        link_n   = n -> bond_idx(n)
        next_matter = n -> (n < N_matter ? n + 1 : 1)
    else
        error("Unknown boundary $boundary")
    end

    os = OpSum()

    if gauge_law === :z2
        # ── Term 1: (1/2) Σ σˣ on each gauge link
        for n in 1:n_links
            os += 0.5, "X", link_n(n)
        end
        # ── Term 2: -(m/2) Σ (-1)^n Z on matter sites (n labelled 1..N_matter)
        for n in 1:N_matter
            sign = (-1)^n
            os += -m / 2 * sign, "Z", matter_idx(n)
        end
        # ── Term 3: (η/4) Σ (X X + Y Y) σ^z   on neighbouring matter pairs
        for n in 1:n_links
            mi  = matter_idx(n)
            bi  = link_n(n)
            mj  = matter_idx(next_matter(n))
            os += η / 4, "X", mi, "Z", bi, "X", mj
            os += η / 4, "Y", mi, "Z", bi, "Y", mj
        end

        # ── Gauss-law SOFT PENALTY for the Z₂-LGT.
        # G_n = σˣ_{n-1,n} · Z_n · σˣ_{n,n+1}, background η_n = (-1)^n.
        # Add -(λ/2) η_n G_n per interior matter site n.  The (λ/2) constant
        # from (λ/2)(1 - η_n G_n) is dropped (doesn't affect spectrum gaps).
        # OBC: drop generators at the two ends (n=1 and n=N_matter).
        # PBC: include all N_matter generators; the left link of n=1 wraps
        # around to bond_idx(N_matter), which is the closing link.
        if boundary === :open_site
            gauss_sites = 2:(N_matter - 1)
        else  # :PBC
            gauss_sites = 1:N_matter
        end
        for n in gauss_sites
            left_link  = boundary === :PBC ? bond_idx(mod1(n - 1, N_matter)) :
                                              bond_idx(n - 1)
            right_link = bond_idx(n)        # PBC: bond_idx(N_matter) = closing link
            ηn = (-1)^n
            os += -λ_gauss / 2 * ηn, "X", left_link, "Z", matter_idx(n), "X", right_link
        end

        # OBC boundary convention switch for the Z₂-LGT Gauss-law penalty.
        #   :drop         -> nothing extra (default, current behaviour)
        #   :truncate_xz  -> treat the missing boundary link as a fixed +1
        #                    background gauge field, giving 2-qubit partial
        #                    Gauss-law generators at the two OBC ends:
        #                      G_1     = Z_{matter_idx(1)}      · σˣ_{link_n(1)}
        #                      G_{N_m} = σˣ_{link_n(N_m - 1)}   · Z_{matter_idx(N_m)}
        #                    penalty -(λ/2) η_n G_n with η_n = (-1)^n.
        if boundary === :open_site && z2_obc_boundary === :truncate_xz
            # G_1 = bg_left · Z_{matter_1} · σˣ_{link_1};  penalty -(λ/2)·η_1·G_1.
            # G_N = σˣ_{link_{N-1}} · Z_{matter_N} · bg_right;  penalty -(λ/2)·η_N·G_N.
            # Default bg_left=bg_right=+1 reproduces the prior code.
            η1 = (-1)^1
            os += -λ_gauss / 2 * η1 * bg_left,  "Z", matter_idx(1),            "X", link_n(1)
            ηN = (-1)^N_matter
            os += -λ_gauss / 2 * ηN * bg_right, "X", link_n(N_matter - 1),     "Z", matter_idx(N_matter)
        end

    elseif gauge_law === :higgs_spt
        # H = g Σ X_link X_site X_link'  + h Σ Z_site Z_link Z_site'  - α Σ X_site
        # with g = h = -1.
        # We center 3-body XXX at each *matter* site so it touches the two
        # adjacent links.  For OBC the two edge matter sites have only one
        # adjacent link — we drop those incomplete triples.
        g = -1.0
        h = -1.0

        # XXX centered on matter site n: X_{link n-1} X_{matter n} X_{link n}
        # Needs both adjacent links to exist.
        for n in 1:N_matter
            left_link = if boundary === :PBC
                # left link of matter n is link between matter (n-1) and n
                link_n(mod1(n - 1, N_matter))
            else
                n == 1 ? nothing : link_n(n - 1)
            end
            right_link = if boundary === :PBC
                link_n(n)
            else
                n == N_matter ? nothing : link_n(n)
            end
            if left_link !== nothing && right_link !== nothing
                os += g, "X", left_link, "X", matter_idx(n), "X", right_link
            end
        end

        # OBC boundary convention switch.  Only used when boundary=:open_site.
        #   :drop        -> nothing extra; canonical cluster model with edge modes
        #   :truncate_xx -> add the truncated XX Gauss-law pieces at the two ends:
        #                     X_{matter 1} · X_{link 1}
        #                     X_{link N-1} · X_{matter N}
        if boundary === :open_site && higgs_obc_boundary === :truncate_xx
            os += g, "X", matter_idx(1),         "X", link_n(1)
            os += g, "X", link_n(N_matter - 1),  "X", matter_idx(N_matter)
        end

        # ZZZ centered on each *link*: Z_{matter n} Z_{link n} Z_{matter n+1}
        for n in 1:n_links
            mi = matter_idx(n)
            bi = link_n(n)
            mj = matter_idx(next_matter(n))
            os += h, "Z", mi, "Z", bi, "Z", mj
        end

        # transverse field on matter sites
        for n in 1:N_matter
            os += -α, "X", matter_idx(n)
        end
    else
        error("Unknown gauge_law $gauge_law (use :z2 or :higgs_spt)")
    end

    return MPO(os, sites)
end

###############################################################################
# 3. DMRG SWEEP SETTINGS
###############################################################################

function dmrg_sweeps(; nsweeps=10, maxdim_final=200, cutoff=1e-10)
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, 20, 40, 80, 120, 160, 200, maxdim_final)
    setcutoff!(sweeps, cutoff)
    setnoise!(sweeps, 1e-6, 1e-6, 1e-7, 1e-8, 0.0)
    return sweeps
end

# For PBC we tend to want a larger χ and a bit more patience
function dmrg_sweeps_pbc(; nsweeps=14, maxdim_final=300, cutoff=1e-10)
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, 20, 40, 80, 120, 160, 200, 250, maxdim_final)
    setcutoff!(sweeps, cutoff)
    setnoise!(sweeps, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0)
    return sweeps
end

###############################################################################
# 4. ENTANGLEMENT SPECTRUM & ENTROPY
###############################################################################

"""
    entanglement_spectrum(psi, bond)

Return (λ, S) at bond `bond`:  Schmidt values (sorted descending,
normalised to unit 2-norm) and the von Neumann entropy.
"""
function entanglement_spectrum(psi::MPS, bond::Int)
    orthogonalize!(psi, bond)

    A  = psi[bond]
    li = linkinds(psi, bond - 1)
    left_inds = [siteind(psi, bond)]
    isempty(li) || push!(left_inds, li...)

    U, S_tensor, V = svd(A, left_inds)
    # S_tensor is a diagonal ITensor with two distinct link indices
    # ("Link,u" and "Link,v") — `diag` returns the singular values directly.
    λ = collect(Float64, diag(S_tensor))
    sort!(λ, rev=true)
    nλ = norm(λ)
    if nλ > 0
        λ ./= nλ
    end
    S = -sum(x^2 * log(x^2 + 1e-16) for x in λ if x > 1e-14)
    return λ, S
end

function full_entanglement_profile(psi::MPS)
    N = length(psi)
    bonds   = 1:(N - 1)
    S_vals  = Float64[]
    λ_list  = Vector{Vector{Float64}}()
    for b in bonds
        λ, S = entanglement_spectrum(psi, b)
        push!(S_vals, S)
        push!(λ_list, λ)
    end
    return collect(bonds), S_vals, λ_list
end

###############################################################################
# 5. ES DEGENERACY DIAGNOSTIC
###############################################################################

"""
    es_degeneracy_check(λ; tol=1e-2)

Count "almost-degenerate" consecutive Schmidt-value pairs.  Returns
(pairs, deg_ratio) where deg_ratio = 2*pairs / length(λ).
"""
function es_degeneracy_check(λ::Vector{Float64}; tol=1e-2)
    pairs = 0
    i = 1
    while i < length(λ)
        if abs(λ[i] - λ[i+1]) / (λ[i] + 1e-16) < tol
            pairs += 1
            i += 2
        else
            i += 1
        end
    end
    deg_ratio = 2pairs / length(λ)
    return pairs, deg_ratio
end

###############################################################################
# 6. STRING ORDER PARAMETER
###############################################################################

"""
    string_order(psi, sites, i, j; op_edge="Z", op_string="X")

Compute  <O_i  ∏_{n=i+1}^{j-1} op_string_n  O_j>.

For PBC we restrict the caller to choose i < j (linear path along the
ring); we deliberately do NOT cross the wrap-around since that doubles
the cost.  See `string_order_profile`.
"""
function string_order(psi::MPS, sites, i::Int, j::Int;
                      op_edge="Z", op_string="X")
    # Build a flat (op, site, op, site, ...) list.  ITensors' OpSum
    # accepts this splatted form to add a single multi-site product
    # operator with a given coefficient.
    args = Any[]
    push!(args, op_edge); push!(args, i)
    for n in (i+1):(j-1)
        push!(args, op_string); push!(args, n)
    end
    push!(args, op_edge); push!(args, j)
    os = OpSum()
    os += (1.0, args...)
    O = MPO(os, sites)
    return inner(psi', O, psi)
end

"""
    string_order_profile(psi, sites; ...)

Average the string order over several (i, i+r) pairs.  For PBC we
restrict separations to r < N/2 to avoid wrapping ambiguity.
"""
function string_order_profile(psi::MPS, sites;
                              op_edge="Z", op_string="X",
                              i_start=3, i_end=nothing, step=2)
    N = length(psi)
    i_end = isnothing(i_end) ? N - 2 : i_end
    r_vals = Int[]
    O_vals = Float64[]
    for i in i_start:step:(N÷2)
        for j in (i+2):step:min(i + N÷2 - 1, N - 1)
            val = real(string_order(psi, sites, i, j;
                                    op_edge=op_edge, op_string=op_string))
            push!(r_vals, j - i)
            push!(O_vals, val)
        end
    end
    r_unique = sort(unique(r_vals))
    O_avg    = [mean(O_vals[r_vals .== r]) for r in r_unique]
    return r_unique, O_avg
end

###############################################################################
# 7. EDGE ENTROPY
###############################################################################

edge_entropy(psi::MPS, n_edge::Int=1) = entanglement_spectrum(psi, n_edge)[2]

###############################################################################
# 8. ENERGY GAP
###############################################################################

"""
    compute_gap(sites, H, sweeps; weight=20.0, χ_init=10)

Returns (E0, E1, gap).
"""
function compute_gap(sites, H, sweeps; weight=20.0, χ_init::Int=10)
    psi0_init = randomMPS(sites, χ_init)
    E0, psi0  = dmrg(H, psi0_init, sweeps; outputlevel=0)
    psi1_init = randomMPS(sites, χ_init)
    E1, psi1  = dmrg(H, [psi0], psi1_init, sweeps;
                     weight=weight, outputlevel=0)
    return E0, E1, E1 - E0, psi0, psi1
end

# Chunked DMRG with early-stopping based on |ΔE| between chunks.  A/B tests
# (test_observer_vs_chunked.jl) found per-sweep cost is ~1.8× cheaper in this
# form than in a single dmrg() with a custom observer, more than offsetting
# the slightly higher sweep count needed.
function _dmrg_chunked(H, psi_init, sweeps_template;
                      prev_states::Union{Nothing,Vector{MPS}}=nothing,
                      weight::Float64=0.0,
                      energy_tol::Float64=0.0,
                      minsweeps::Int=20,
                      chunk_sweeps::Int=10)
    n_total = ITensorMPS.nsweep(sweeps_template)
    # If early-stop disabled, just defer to the original single dmrg() call.
    if energy_tol <= 0
        if prev_states === nothing
            return dmrg(H, psi_init, sweeps_template; outputlevel=0)
        else
            return dmrg(H, prev_states, psi_init, sweeps_template;
                        weight=weight, outputlevel=0)
        end
    end
    # Build a chunk-sized Sweeps that mirrors the template's per-sweep
    # settings.  Use the per-sweep accessors maxdim/cutoff/noise(sw, i)
    # since there's no bulk-array getter on Sweeps.
    md_arr = [maxdim(sweeps_template, i) for i in 1:n_total]
    co_arr = [cutoff(sweeps_template, i) for i in 1:n_total]
    no_arr = [noise(sweeps_template, i)  for i in 1:n_total]
    function _chunk_sweeps(n)
        s = Sweeps(n)
        setmaxdim!(s, md_arr[1:n]...)
        setcutoff!(s, co_arr[1:n]...)
        setnoise!(s, no_arr[1:n]...)
        return s
    end
    psi = psi_init
    E   = Inf
    swept = 0
    while swept < n_total
        nsw_this = min(chunk_sweeps, n_total - swept)
        sw = _chunk_sweeps(nsw_this)
        E_new, psi = prev_states === nothing ?
            dmrg(H, psi, sw; outputlevel=0) :
            dmrg(H, prev_states, psi, sw; weight=weight, outputlevel=0)
        swept += nsw_this
        dE = abs(E_new - E)
        E = E_new
        if swept >= minsweeps && isfinite(dE) && dE < energy_tol
            break
        end
    end
    return E, psi
end

"""
    compute_excited_states(sites, H, sweeps; k=4, weight=30.0, χ_init=10)

Return (energies::Vector{Float64}, states::Vector{MPS}) for E₀..E_k via
sequential orthogonality-weight DMRG.  Each excited state DMRG is run
with all previously found states as the orthogonal-set with `weight`
penalty.

`sweeps` is the sweep schedule used for E₀.  For higher excited states
we use a slightly longer schedule (more sweeps, larger maxdim) since
they are typically harder to converge.
"""
function compute_excited_states(sites, H, sweeps; k::Int=4,
                                weight::Float64=30.0, χ_init::Int=10,
                                higher_sweeps=nothing,
                                early_stop_tol::Float64=0.0,
                                minsweeps::Int=20,
                                chunk_sweeps::Int=10)
    energies = Float64[]
    states   = MPS[]

    _t_start = time()
    psi0_init = randomMPS(sites, χ_init)
    _t_gs = time()
    E0, psi0 = _dmrg_chunked(H, psi0_init, sweeps;
                             weight=0.0, energy_tol=early_stop_tol,
                             minsweeps=minsweeps, chunk_sweeps=chunk_sweeps)
    push!(energies, E0)
    push!(states,   psi0)
    @printf("  [compute_excited] k=0/%d  E=%.5f  Δt=%.1fs  total=%.1fs\n",
            k, E0, time()-_t_gs, time()-_t_start); flush(stdout)

    hs = higher_sweeps === nothing ? sweeps : higher_sweeps

    for j in 1:k
        _t_j = time()
        psi_init = randomMPS(sites, χ_init)
        Ej, psij = _dmrg_chunked(H, psi_init, hs;
                                 prev_states=states,
                                 weight=weight, energy_tol=early_stop_tol,
                                 minsweeps=minsweeps, chunk_sweeps=chunk_sweeps)
        push!(energies, Ej)
        push!(states,   psij)
        _elapsed = time() - _t_start
        _avg = _elapsed / (j + 1)
        _eta = max(0.0, _avg * (k - j))
        @printf("  [compute_excited] k=%d/%d  E=%.5f  gap=%.5f  Δt=%.1fs  total=%.1fs  ETA=%.1fs\n",
                j, k, Ej, Ej - energies[1], time()-_t_j, _elapsed, _eta)
        flush(stdout)
    end
    return energies, states
end

###############################################################################
# 8a. PER-STATE QUANTUM-NUMBER DIAGNOSTICS
###############################################################################

"""
    state_quantum_numbers(psi, sites, boundary)

Return a NamedTuple with diagnostic quantum numbers used to label
eigenstates of the Z₂-LGT Hamiltonian:

  - `matter_parity`   = ⟨ψ| ∏_n Z_{matter_idx(n)} |ψ⟩   (expected ±1)
  - `total_Z_matter`  = ⟨ψ| Σ_n Z_{matter_idx(n)} |ψ⟩
  - `total_X_link`    = ⟨ψ| Σ_l σˣ_{link_n(l)} |ψ⟩

`boundary` is `:open_site` (N_total = 2N_matter−1, n_links = N_matter−1)
or `:PBC` (N_total = 2N_matter, n_links = N_matter).
"""
function state_quantum_numbers(psi::MPS, sites, boundary::Symbol)
    N_total = length(sites)
    if boundary === :open_site
        @assert isodd(N_total)
        N_matter = (N_total + 1) ÷ 2
        n_links  = N_matter - 1
    elseif boundary === :PBC
        @assert iseven(N_total)
        N_matter = N_total ÷ 2
        n_links  = N_matter
    else
        error("Unknown boundary $boundary")
    end

    matter_inds = [matter_idx(n) for n in 1:N_matter]
    link_inds   = [bond_idx(l)   for l in 1:n_links]

    # ── Matter parity:  ⟨∏_n Z_{matter_idx(n)}⟩
    # Build the global Z-string MPO via OpSum and use inner(psi', O, psi).
    args = Any[]
    for mi in matter_inds
        push!(args, "Z"); push!(args, mi)
    end
    os = OpSum()
    os += (1.0, args...)
    Ostring = MPO(os, sites)
    matter_parity = real(inner(psi', Ostring, psi))

    # ── Sums of single-site expectations.
    Zvals = expect(psi, "Z")
    Xvals = expect(psi, "X")
    total_Z_matter = sum(real(Zvals[mi]) for mi in matter_inds)
    total_X_link   = sum(real(Xvals[li]) for li in link_inds)

    return (matter_parity   = matter_parity,
            total_Z_matter  = total_Z_matter,
            total_X_link    = total_X_link)
end

###############################################################################
# 9. CENTRAL CHARGE FIT
###############################################################################

function fit_central_charge(bonds::Vector{Int}, S_vals::Vector{Float64}, N::Int)
    xs = [log(sin(π * b / N) + 1e-14) for b in bonds]
    n  = length(xs)
    sx = sum(xs);  sy = sum(S_vals)
    sxx = sum(xs .^ 2);  sxy = sum(xs .* S_vals)
    α = (n * sxy - sx * sy) / (n * sxx - sx^2)
    β = (sy - α * sx) / n
    c = 3α
    return c, α, β
end

###############################################################################
# 10. VIRTUAL-SPACE SYMMETRY REP
###############################################################################

function virtual_symmetry_rep(psi::MPS, sites, bond::Int;
                              g_ops::Vector{String}=["X"])
    N = length(psi)
    orthogonalize!(psi, bond)
    L_env = ITensor(1.0)
    for i in 1:bond
        s  = sites[i]
        gi = op(s, g_ops[mod1(i, length(g_ops))])
        A  = psi[i]
        gA = noprime(gi * prime(A, s))
        if i == 1
            L_env = gA * dag(psi[1])
        else
            L_env = L_env * gA * dag(psi[i])
        end
    end
    link_l = linkind(psi, bond)
    Ug = matrix(L_env, link_l, dag(link_l)')
    return Ug
end

###############################################################################
# 11. PHASE-DIAGRAM SCAN (generic)
###############################################################################

function phase_scan(N_matter::Int, m_vals::Vector{Float64}, η_vals::Vector{Float64};
                    boundary::Symbol=:open_site, gauge_law::Symbol=:z2,
                    α::Float64=1.0, maxdim::Int=100, cutoff::Float64=1e-8)
    sites  = build_sites(N_matter; boundary=boundary)
    N_tot  = length(sites)
    mid    = N_tot ÷ 2
    sweeps = boundary === :PBC ? dmrg_sweeps_pbc(maxdim_final=maxdim, cutoff=cutoff) :
                                 dmrg_sweeps(maxdim_final=maxdim, cutoff=cutoff)

    nm, nη = length(m_vals), length(η_vals)
    E0_arr      = zeros(nm, nη)
    gap_arr     = zeros(nm, nη)
    S_mid_arr   = zeros(nm, nη)
    deg_arr     = zeros(nm, nη)
    Ostr_arr    = zeros(nm, nη)
    S_edge_arr  = zeros(nm, nη)
    phase_arr   = Matrix{Symbol}(undef, nm, nη)

    for (im, m) in enumerate(m_vals), (iη, η) in enumerate(η_vals)
        @printf("  m=%.3f  η=%.3f\n", m, η)
        H    = build_hamiltonian(sites, m, η;
                                 boundary=boundary, gauge_law=gauge_law, α=α)
        E0, E1, gap, psi0, _ = compute_gap(sites, H, sweeps)

        λ_mid, S_mid = entanglement_spectrum(psi0, mid)
        _, dratio    = es_degeneracy_check(λ_mid)
        i_s, j_s     = max(1, mid - 6), min(N_tot, mid + 6)
        Ostr         = real(string_order(psi0, sites, i_s, j_s))
        S_edge       = edge_entropy(psi0, 1)

        E0_arr[im,iη]     = E0
        gap_arr[im,iη]    = gap
        S_mid_arr[im,iη]  = S_mid
        deg_arr[im,iη]    = dratio
        Ostr_arr[im,iη]   = Ostr
        S_edge_arr[im,iη] = S_edge
        phase_arr[im,iη]  = classify_phase(gap, dratio, S_edge, Ostr;
                                           boundary=boundary)
    end

    return (m=m_vals, η=η_vals,
            E0=E0_arr, gap=gap_arr, S_mid=S_mid_arr,
            deg_ratio=deg_arr, string_order=Ostr_arr,
            S_edge=S_edge_arr, phase=phase_arr)
end

###############################################################################
# 12. FINITE-SIZE SCALING OF GAP
###############################################################################

function gap_finite_size_scaling(m::Float64, η::Float64, N_list::Vector{Int};
                                  boundary::Symbol=:open_site,
                                  gauge_law::Symbol=:z2, α::Float64=1.0,
                                  maxdim::Int=150, cutoff::Float64=1e-10)
    gaps = Float64[]
    for N in N_list
        sites = build_sites(N; boundary=boundary)
        H = build_hamiltonian(sites, m, η;
                              boundary=boundary, gauge_law=gauge_law, α=α)
        sw = boundary === :PBC ? dmrg_sweeps_pbc(maxdim_final=maxdim, cutoff=cutoff) :
                                 dmrg_sweeps(maxdim_final=maxdim, cutoff=cutoff)
        _, _, gap, _, _ = compute_gap(sites, H, sw)
        push!(gaps, gap)
        @printf("  N=%d  gap=%.6e\n", N, gap)
    end
    log_gaps = log.(max.(gaps, 1e-14))
    n  = length(N_list)
    sx = sum(N_list);   sy = sum(log_gaps)
    sxx = sum(N_list .^ 2); sxy = sum(N_list .* log_gaps)
    αslope  = (n * sxy - sx * sy) / (n * sxx - sx^2)
    ξ = -1.0 / αslope
    @printf("  Fitted correlation length ξ = %.3f\n", ξ)
    return N_list, gaps, ξ
end

###############################################################################
# 13. PHASE CLASSIFICATION + diagnose() WRAPPER
#
# Threshold choices (DOCUMENTED HERE — the other implementation may choose
# differently; report these to the cross-check):
#   * gapless     : gap < 0.05   (in units of the Hamiltonian)
#   * SPT_like    : gapped AND  deg_ratio > 0.5 AND S_edge > 0.5*log(2)
#   * topological : gapped AND  deg_ratio > 0.5 AND NOT string-order
#   * SPT_full    : SPT_like AND |Ostr| > 0.05
#   * trivial     : otherwise
###############################################################################

const GAP_THRESHOLD     = 0.05
const DEG_THRESHOLD     = 0.5
const EDGE_THRESHOLD    = 0.5 * log(2)
const STRING_THRESHOLD  = 0.05

function classify_phase(gap, deg_ratio, S_edge, Ostr; boundary::Symbol=:open_site)
    if gap < GAP_THRESHOLD
        return :gapless_or_critical
    end
    edge_sig   = (boundary === :open_site) && (S_edge > EDGE_THRESHOLD)
    topo_es    = deg_ratio > DEG_THRESHOLD
    string_sig = abs(Ostr) > STRING_THRESHOLD
    if topo_es && string_sig && edge_sig
        return :SPT
    elseif topo_es && edge_sig
        return :topological
    elseif topo_es && string_sig
        # PBC version: no edge to check
        return :SPT_bulk
    else
        return :trivial_gapped
    end
end

"""
    diagnose(psi, sites; boundary=:open_site, gauge_law=:z2, gap=nothing)

Run the suite of diagnostics on a given MPS.  Returns a NamedTuple with
`gap`, `S_mid`, `deg_ratio`, `Ostr`, `S_edge`, `phase`.

`gap` is taken as input (since computing it requires running another
DMRG and we usually already have one).  If `nothing` is passed in, it
is returned as NaN and the phase classification falls back accordingly.
"""
function diagnose(psi::MPS, sites; boundary::Symbol=:open_site,
                  gauge_law::Symbol=:z2, gap=nothing)
    N_tot = length(psi)
    mid   = N_tot ÷ 2
    λ_mid, S_mid = entanglement_spectrum(psi, mid)
    _, dratio    = es_degeneracy_check(λ_mid)

    # String-order range: pick a path of length ≈ N/3 to stay away from
    # edges (OBC) or from the wrap point (PBC).
    if boundary === :open_site
        i_s = max(1, mid - N_tot÷4)
        j_s = min(N_tot, mid + N_tot÷4)
    else  # PBC
        # restrict to one half of the ring (avoid wrap)
        i_s = max(1, mid - N_tot÷5)
        j_s = min(N_tot, mid + N_tot÷5)
    end
    Ostr_zx = real(string_order(psi, sites, i_s, j_s; op_edge="Z", op_string="X"))
    Ostr_xz = real(string_order(psi, sites, i_s, j_s; op_edge="X", op_string="Z"))
    Ostr    = abs(Ostr_zx) > abs(Ostr_xz) ? Ostr_zx : Ostr_xz

    if boundary === :open_site
        S_edge = edge_entropy(psi, 1)
    else
        # No physical edge on a ring; reuse bond-1 entropy for reporting only.
        S_edge = edge_entropy(psi, 1)
    end

    gap_val = gap === nothing ? NaN : gap
    phase   = classify_phase(gap_val, dratio, S_edge, Ostr; boundary=boundary)

    return (gap=gap_val, S_mid=S_mid, deg_ratio=dratio,
            Ostr=Ostr, Ostr_ZX=Ostr_zx, Ostr_XZ=Ostr_xz,
            S_edge=S_edge, phase=phase)
end

###############################################################################
# 14. SECTION III SMOKE RUN
#
# Reference point from the paper:  m_0 = 1.125, η = 1.0, α = 1.0
# L (a.k.a. N_matter) ∈ {4, 6, 8}; boundary ∈ {:open_site, :PBC};
# gauge_law ∈ {:z2, :higgs_spt}.
###############################################################################

const DATA_DIR = joinpath(@__DIR__, "data", "z2_claude")

function _ensure_data_dir()
    isdir(DATA_DIR) || mkpath(DATA_DIR)
end

"""
    section_iii_smoke_run(; N_list=[4,6,8], m=1.125, η=1.0, α=1.0)

Compute gap and diagnostics for all (boundary, gauge_law) × L combinations
at the paper's reference point.  Writes two CSVs into DATA_DIR.
"""
function section_iii_smoke_run(; N_list::Vector{Int}=[4, 6, 8],
                                m::Float64=1.125, η::Float64=1.0,
                                α::Float64=1.0,
                                higgs_obc_boundary::Symbol=:drop)
    _ensure_data_dir()
    @printf("\n[smoke] higgs_obc_boundary=%s\n", higgs_obc_boundary)

    boundaries = (:open_site, :PBC)
    gauges     = (:z2, :higgs_spt)

    # 1. Gap table:  L, boundary, gauge_law, E0, E1, gap
    rows_gap = String[]
    push!(rows_gap, "L,boundary,gauge_law,E0,E1,gap,phase,S_mid,deg_ratio,Ostr,S_edge")
    # 2. Ratio table:  L, gauge_law, gap_OBC, gap_PBC, ratio
    rows_ratio = String[]
    push!(rows_ratio, "L,gauge_law,gap_OBC,gap_PBC,ratio_OBC_over_PBC")

    # Store gaps so we can compute the ratio
    gap_store = Dict{Tuple{Int,Symbol,Symbol}, Float64}()
    diag_store = Dict{Tuple{Int,Symbol,Symbol}, NamedTuple}()

    for L in N_list, b in boundaries, gl in gauges
        sites = build_sites(L; boundary=b)
        H     = build_hamiltonian(sites, m, η;
                                  boundary=b, gauge_law=gl, α=α,
                                  higgs_obc_boundary=higgs_obc_boundary)
        sw    = b === :PBC ? dmrg_sweeps_pbc(maxdim_final=300, cutoff=1e-10) :
                             dmrg_sweeps(maxdim_final=200, cutoff=1e-10)
        @printf("\n[smoke] L=%d  boundary=%s  gauge_law=%s\n", L, b, gl)
        E0, E1, gap, psi0, _ = compute_gap(sites, H, sw)
        @printf("        E0=%.8f  E1=%.8f  gap=%.6e\n", E0, E1, gap)

        d = diagnose(psi0, sites; boundary=b, gauge_law=gl, gap=gap)

        gap_store[(L, b, gl)]  = gap
        diag_store[(L, b, gl)] = d

        push!(rows_gap, @sprintf("%d,%s,%s,%.8f,%.8f,%.6e,%s,%.4f,%.4f,%.6f,%.4f",
                                 L, b, gl, E0, E1, gap, d.phase,
                                 d.S_mid, d.deg_ratio, d.Ostr, d.S_edge))
    end

    # Build ratio rows
    for L in N_list, gl in gauges
        g_o = get(gap_store, (L, :open_site, gl), NaN)
        g_p = get(gap_store, (L, :PBC,       gl), NaN)
        ratio = (isfinite(g_o) && isfinite(g_p) && abs(g_p) > 1e-12) ?
                g_o / g_p : NaN
        push!(rows_ratio, @sprintf("%d,%s,%.6e,%.6e,%.6f", L, gl, g_o, g_p, ratio))
    end

    suffix = "_higgs_$(higgs_obc_boundary)"
    fname_table = "section_iii_gap_table$(suffix).csv"
    fname_ratio = "section_iii_gap_ratio$(suffix).csv"
    open(joinpath(DATA_DIR, fname_table), "w") do io
        for r in rows_gap; println(io, r); end
    end
    open(joinpath(DATA_DIR, fname_ratio), "w") do io
        for r in rows_ratio; println(io, r); end
    end

    println("\n[smoke] Wrote ", joinpath(DATA_DIR, fname_table))
    println("[smoke] Wrote ",   joinpath(DATA_DIR, fname_ratio))
    return gap_store, diag_store
end

"""
    convergence_run(; N_list=[4,6,8,10,12], k_max=4,
                      m=1.125, η=1.0, α=1.0,
                      z2_obc_boundary=:drop)

For each N_matter in `N_list` and each boundary ∈ {:open_site, :PBC},
compute the first `k_max+1` eigenstates (E₀..E_{k_max}) of the Z₂-LGT
Hamiltonian at the paper's reference point (m=1.125, η=1.0, α=1.0).
For each eigenstate compute the three quantum-number diagnostics
(`matter_parity`, `total_Z_matter`, `total_X_link`).

Writes one CSV:
    DATA_DIR/convergence_<z2_obc_boundary>.csv
with columns
    L, boundary, z2_obc_boundary, k, E_k, gap_k,
      matter_parity, total_Z_matter, total_X_link
"""
function convergence_run(; N_list::Vector{Int}=[4, 6, 8, 10, 12],
                          k_max::Int=4,
                          m::Float64=1.125, η::Float64=1.0,
                          α::Float64=1.0,
                          z2_obc_boundary::Symbol=:drop)
    _ensure_data_dir()
    @printf("\n[convergence] z2_obc_boundary=%s  k_max=%d\n",
            z2_obc_boundary, k_max)

    boundaries = (:open_site, :PBC)

    rows = String[]
    push!(rows, "L,boundary,z2_obc_boundary,k,E_k,gap_k,matter_parity,total_Z_matter,total_X_link")

    for L in N_list, b in boundaries
        @printf("\n[convergence] L=%d  boundary=%s\n", L, b)
        sites = build_sites(L; boundary=b)
        H     = build_hamiltonian(sites, m, η;
                                  boundary=b, gauge_law=:z2, α=α,
                                  z2_obc_boundary=z2_obc_boundary)

        # Use a more aggressive schedule for the excited-state runs.
        if b === :PBC
            sweeps_gs     = dmrg_sweeps_pbc(nsweeps=14, maxdim_final=300, cutoff=1e-10)
            sweeps_higher = dmrg_sweeps_pbc(nsweeps=18, maxdim_final=400, cutoff=1e-10)
        else
            sweeps_gs     = dmrg_sweeps(nsweeps=12, maxdim_final=250, cutoff=1e-10)
            sweeps_higher = dmrg_sweeps(nsweeps=16, maxdim_final=350, cutoff=1e-10)
        end

        energies, states = compute_excited_states(sites, H, sweeps_gs;
                                                  k=k_max, weight=30.0,
                                                  χ_init=10,
                                                  higher_sweeps=sweeps_higher)

        E0 = energies[1]
        for (k_idx, Ek) in enumerate(energies)
            k = k_idx - 1   # k = 0..k_max
            qn = state_quantum_numbers(states[k_idx], sites, b)
            gap_k = Ek - E0
            @printf("  k=%d  E_k=%.8f  gap_k=%.6e  P=%+.4f  ΣZ=%+.4f  ΣX=%+.4f\n",
                    k, Ek, gap_k, qn.matter_parity,
                    qn.total_Z_matter, qn.total_X_link)
            push!(rows, @sprintf("%d,%s,%s,%d,%.10f,%.10f,%.6f,%.6f,%.6f",
                                  L, b, z2_obc_boundary, k,
                                  Ek, gap_k,
                                  qn.matter_parity,
                                  qn.total_Z_matter,
                                  qn.total_X_link))
        end
    end

    fname = @sprintf("convergence_%s.csv", z2_obc_boundary)
    open(joinpath(DATA_DIR, fname), "w") do io
        for r in rows; println(io, r); end
    end
    println("\n[convergence] Wrote ", joinpath(DATA_DIR, fname))
    return nothing
end

"""
    phase_scan_smoke(; N_matter=4, boundary=:open_site, gauge_law=:z2, α=1.0)

Small (m_0, η) grid at one L and one boundary, classifying each point.
Writes a CSV into DATA_DIR.
"""
function phase_scan_smoke(; N_matter::Int=4,
                            boundary::Symbol=:open_site,
                            gauge_law::Symbol=:z2,
                            α::Float64=1.0,
                            m_vals::Vector{Float64}=[0.0, 0.5, 1.0, 1.5],
                            η_vals::Vector{Float64}=[0.5, 1.0, 1.5])
    _ensure_data_dir()
    @printf("[phase_scan] L=%d  boundary=%s  gauge_law=%s\n",
            N_matter, boundary, gauge_law)
    result = phase_scan(N_matter, m_vals, η_vals;
                        boundary=boundary, gauge_law=gauge_law, α=α,
                        maxdim=120, cutoff=1e-9)

    rows = String[]
    push!(rows, "L,boundary,gauge_law,m,eta,E0,gap,S_mid,deg_ratio,Ostr,S_edge,phase")
    for (im, m) in enumerate(m_vals), (iη, η) in enumerate(η_vals)
        push!(rows, @sprintf("%d,%s,%s,%.3f,%.3f,%.8f,%.6e,%.4f,%.4f,%.6f,%.4f,%s",
                              N_matter, boundary, gauge_law, m, η,
                              result.E0[im,iη],     result.gap[im,iη],
                              result.S_mid[im,iη],  result.deg_ratio[im,iη],
                              result.string_order[im,iη],
                              result.S_edge[im,iη], result.phase[im,iη]))
    end
    fname = @sprintf("phase_scan_L%d_%s_%s.csv", N_matter, boundary, gauge_law)
    open(joinpath(DATA_DIR, fname), "w") do io
        for r in rows; println(io, r); end
    end
    println("[phase_scan] Wrote ", joinpath(DATA_DIR, fname))
    return result
end

###############################################################################
# 14b. SECTOR-FILTERED MESON GAP (Option 1)
#
# Run excited-state DMRG to k_max (default 12).  For each (N_matter, boundary,
# z2_obc_boundary), find the lowest k >= 1 whose (matter_parity, total_Z_matter)
# agree with the ground state (within tol):
#   |P_k - P_0| <= P_tol         (default 0.05 — matter parity is ±1)
#   |total_Z_matter_k| <= Z_tol  (default 0.5 — charge-neutral)
# This identifies the same-(P, ΣZ) sector — the meson candidate.
# If none found, k_meson = -1, gap_meson = NaN.
###############################################################################

"""
    convergence_sector_filtered(; N_list=[4,6,8,10,12], k_max=12,
                                  m=1.125, η=1.0, α=1.0,
                                  P_tol=0.05, Z_tol=0.5)

For each (L, boundary, z2_obc_boundary), do k_max excited-state DMRG runs,
then pick the lowest k≥1 that shares the ground-state (matter_parity,
total_Z_matter) sector (within tolerances).  Write a single CSV
`convergence_sector.csv` with one row per (L, boundary, z2_obc_boundary).
"""
function convergence_sector_filtered(; N_list::Vector{Int}=[4, 6, 8, 10, 12],
                                       k_max::Int=12,
                                       m::Float64=1.125, η::Float64=1.0,
                                       α::Float64=1.0,
                                       P_tol::Float64=0.05,
                                       Z_tol::Float64=0.5)
    _ensure_data_dir()
    @printf("\n[sector] sector-filtered meson gap, k_max=%d\n", k_max)

    rows = String[]
    push!(rows, "L,boundary,z2_obc_boundary,k_meson,gap_meson,P_meson,totalZ_meson,totalX_meson,k_max_searched")

    # The (boundary, z2_obc_boundary) combinations to scan.
    # For PBC, z2_obc_boundary is meaningless — we report it as "n/a"
    # to keep one canonical row per (L, :PBC).
    cases = [
        (:open_site, :drop),
        (:open_site, :truncate_xz),
        (:PBC,       :drop),       # one PBC row
    ]

    for L in N_list, (b, z2obc) in cases
        z2obc_label = b === :PBC ? "n/a" : String(z2obc)
        @printf("\n[sector] L=%d  boundary=%s  z2_obc=%s\n", L, b, z2obc_label)
        sites = build_sites(L; boundary=b)
        H     = build_hamiltonian(sites, m, η;
                                  boundary=b, gauge_law=:z2, α=α,
                                  z2_obc_boundary=z2obc)

        # Generous schedule — k_max=12 with weight orthogonality is expensive.
        if b === :PBC
            sweeps_gs     = dmrg_sweeps_pbc(nsweeps=14, maxdim_final=300, cutoff=1e-10)
            sweeps_higher = dmrg_sweeps_pbc(nsweeps=18, maxdim_final=400, cutoff=1e-10)
        else
            sweeps_gs     = dmrg_sweeps(nsweeps=12, maxdim_final=250, cutoff=1e-10)
            sweeps_higher = dmrg_sweeps(nsweeps=16, maxdim_final=400, cutoff=1e-10)
        end

        # Collect energies & quantum numbers
        energies = Float64[]
        qns = NamedTuple[]
        try
            es, sts = compute_excited_states(sites, H, sweeps_gs;
                                              k=k_max, weight=50.0,
                                              χ_init=10,
                                              higher_sweeps=sweeps_higher)
            for (idx, Ek) in enumerate(es)
                push!(energies, Ek)
                push!(qns, state_quantum_numbers(sts[idx], sites, b))
            end
        catch err
            @warn "DMRG run failed at L=$L boundary=$b z2_obc=$z2obc_label: $err"
        end

        if isempty(energies)
            push!(rows, @sprintf("%d,%s,%s,%d,%.10f,%.6f,%.6f,%.6f,%d",
                                  L, b, z2obc_label, -1, NaN, NaN, NaN, NaN, k_max))
            continue
        end

        E0 = energies[1]
        P0 = qns[1].matter_parity
        # Find lowest k>=1 in same sector
        k_meson = -1
        for k in 2:length(energies)   # k_idx, so physical k = k_idx-1; we want k>=1 i.e. k_idx>=2
            P_k = qns[k].matter_parity
            Z_k = qns[k].total_Z_matter
            if abs(P_k - P0) <= P_tol && abs(Z_k) <= Z_tol
                k_meson = k - 1
                break
            end
        end

        if k_meson < 0
            @printf("  No k<=%d satisfies the sector filter; reporting -1.\n", k_max)
            push!(rows, @sprintf("%d,%s,%s,%d,%.10f,%.6f,%.6f,%.6f,%d",
                                  L, b, z2obc_label, -1, NaN, NaN, NaN, NaN, k_max))
        else
            kidx = k_meson + 1
            gap  = energies[kidx] - E0
            qn   = qns[kidx]
            @printf("  k_meson=%d  gap_meson=%.6e  P=%+.4f  ΣZ=%+.4f  ΣX=%+.4f\n",
                    k_meson, gap, qn.matter_parity, qn.total_Z_matter, qn.total_X_link)
            push!(rows, @sprintf("%d,%s,%s,%d,%.10f,%.6f,%.6f,%.6f,%d",
                                  L, b, z2obc_label, k_meson, gap,
                                  qn.matter_parity, qn.total_Z_matter, qn.total_X_link, k_max))
        end
    end

    fname = "convergence_sector.csv"
    open(joinpath(DATA_DIR, fname), "w") do io
        for r in rows; println(io, r); end
    end
    println("\n[sector] Wrote ", joinpath(DATA_DIR, fname))
    return nothing
end

###############################################################################
# 14c. OVERLAP-TARGETED MESON GAP (Option 2)
#
# Compute |Ω⟩ ≈ ψ_0 (ground state). Build the 3-site MPO
#     O_center = X_{matter_idx(m)} · Z_{bond_idx(m)} · X_{matter_idx(m+1)}
# with m = N_matter ÷ 2 (so the operator is centered).  Then for each ψ_k
# in 0..K (with K=8) compute |⟨ψ_k|O_center|ψ_0⟩|².  The "meson eigenstate"
# is the k>0 (or k=0 if it dominates) with the largest overlap.
###############################################################################

"""
    build_O_center_mpo(sites, N_matter, boundary)

Build the 3-site operator O_center = X_{matter_idx(m)} · Z_{bond_idx(m)} ·
X_{matter_idx(m+1)} as an MPO on `sites`, where m = N_matter ÷ 2.
"""
function build_O_center_mpo(sites, N_matter::Int, boundary::Symbol)
    m_center = N_matter ÷ 2
    @assert m_center >= 1 "Need N_matter >= 2 for the centered O_center."
    # For PBC, matter site (m_center+1) is well-defined when m_center+1 <= N_matter
    # Both OBC (chain length 2N-1) and PBC (chain length 2N) have matter_idx(m+1)
    # within range for m_center+1 <= N_matter.
    mi  = matter_idx(m_center)
    bi  = bond_idx(m_center)
    mj  = matter_idx(m_center + 1)
    os = OpSum()
    os += 1.0, "X", mi, "Z", bi, "X", mj
    return MPO(os, sites), m_center
end

"""
    build_O_pzero_mpo(sites, N_matter, boundary) -> (MPO, n_terms)

Momentum-zero meson creation operator:
    O_p=0 = (1/√n_terms) Σ_n X_{matter n} · σᶻ_{n,n+1} · X_{matter n+1}
OBC sums n=1..N_matter-1 (n_terms = N_matter-1). PBC adds wrap-around
n=N_matter (closing link), n_terms = N_matter.
"""
function build_O_pzero_mpo(sites, N_matter::Int, boundary::Symbol)
    n_terms = boundary === :PBC ? N_matter : (N_matter - 1)
    norm = 1.0 / sqrt(n_terms)
    os = OpSum()
    for n in 1:(N_matter - 1)
        os += norm, "X", matter_idx(n), "Z", bond_idx(n), "X", matter_idx(n + 1)
    end
    if boundary === :PBC
        os += norm, "X", matter_idx(N_matter), "Z", bond_idx(N_matter), "X", matter_idx(1)
    end
    return MPO(os, sites), n_terms
end

"""
    convergence_overlap(; N_list=[4,6,8,10,12], K=8,
                          m=1.125, η=1.0, α=1.0)

For each (L, boundary, z2_obc_boundary) compute ψ_0..ψ_K and the overlap
|⟨ψ_k | O_center | ψ_0⟩|² for k = 0..K.  Report (k_dominant, gap_dominant,
overlap_dominant) with k_dominant chosen as argmax over k=0..K (the prompt
allows k=0 to dominate).
"""
function convergence_overlap(; N_list::Vector{Int}=[4, 6, 8, 10, 12],
                              K::Int=8,
                              m::Float64=1.125, η::Float64=1.0,
                              α::Float64=1.0)
    _ensure_data_dir()
    @printf("\n[overlap] |P⟩=O_center|Ω⟩ overlap targeting, K=%d\n", K)

    cases = [
        (:open_site, :drop),
        (:open_site, :truncate_xz),
        (:PBC,       :drop),
    ]

    rows = String[]
    # header: L,boundary,z2_obc_boundary,k_dominant,gap_dominant,overlap_dominant,overlap_k0,...,overlap_kK
    overlap_cols = join(["overlap_k$(k)" for k in 0:K], ",")
    push!(rows, "L,boundary,z2_obc_boundary,k_dominant,gap_dominant,overlap_dominant,$(overlap_cols)")

    for L in N_list, (b, z2obc) in cases
        z2obc_label = b === :PBC ? "n/a" : String(z2obc)
        @printf("\n[overlap] L=%d  boundary=%s  z2_obc=%s\n", L, b, z2obc_label)
        sites = build_sites(L; boundary=b)
        H     = build_hamiltonian(sites, m, η;
                                  boundary=b, gauge_law=:z2, α=α,
                                  z2_obc_boundary=z2obc)

        if b === :PBC
            sweeps_gs     = dmrg_sweeps_pbc(nsweeps=14, maxdim_final=300, cutoff=1e-10)
            sweeps_higher = dmrg_sweeps_pbc(nsweeps=18, maxdim_final=400, cutoff=1e-10)
        else
            sweeps_gs     = dmrg_sweeps(nsweeps=12, maxdim_final=250, cutoff=1e-10)
            sweeps_higher = dmrg_sweeps(nsweeps=16, maxdim_final=400, cutoff=1e-10)
        end

        energies = Float64[]
        states   = MPS[]
        try
            es, sts = compute_excited_states(sites, H, sweeps_gs;
                                              k=K, weight=50.0,
                                              χ_init=10,
                                              higher_sweeps=sweeps_higher)
            energies = es
            states   = sts
        catch err
            @warn "DMRG failed at L=$L boundary=$b z2_obc=$z2obc_label: $err"
            row = @sprintf("%d,%s,%s,%d,%.10f,%.10f", L, b, z2obc_label, -1, NaN, NaN)
            row *= "," * join([@sprintf("%.10f", NaN) for _ in 0:K], ",")
            push!(rows, row)
            continue
        end

        E0 = energies[1]
        psi0 = states[1]

        # Build O_center MPO
        O_mpo, m_center = build_O_center_mpo(sites, L, b)
        @printf("  m_center=%d (chain indices: %d, %d, %d)\n",
                m_center, matter_idx(m_center), bond_idx(m_center),
                matter_idx(m_center + 1))

        # Compute |P⟩ = O psi0 (just to report normalization);
        # for the overlap we use inner(ψ_k', O, ψ_0) directly.
        # Norm of |P⟩ = sqrt(<ψ0|O† O|ψ0>) = 1 because each operator is unitary
        # (X, Z are involutions, so O is a product of involutions on
        # disjoint qubits → O is itself unitary).  Hence |⟨ψ_k|P⟩|² is the
        # squared overlap with the *normalised* |P⟩ directly.
        overlaps = Float64[]
        for k in 0:K
            kidx = k + 1
            if kidx > length(states)
                push!(overlaps, NaN)
            else
                amp = inner(states[kidx]', O_mpo, psi0)
                push!(overlaps, abs2(amp))
            end
        end

        # Find dominant k (argmax over all k; the prompt says don't exclude k=0
        # unless k=0 dominates — possible — so just argmax).
        valid = findall(!isnan, overlaps)
        if isempty(valid)
            k_dominant   = -1
            gap_dominant = NaN
            ov_dominant  = NaN
        else
            kbest = valid[1]
            for kk in valid
                if overlaps[kk] > overlaps[kbest]
                    kbest = kk
                end
            end
            k_dominant   = kbest - 1
            gap_dominant = energies[kbest] - E0
            ov_dominant  = overlaps[kbest]
        end

        # Verify operator norm via |P⟩ norm — useful diagnostic.
        # |P⟩ = O|ψ0⟩ ⇒ ⟨P|P⟩ = ⟨ψ0|O†O|ψ0⟩ = inner(O, ψ0, O, ψ0).
        Pnorm  = sqrt(max(real(inner(O_mpo, psi0, O_mpo, psi0)), 0.0))
        @printf("  |P⟩ norm = %.6f (should be ≈1 since O = X·Z·X is unitary)\n", Pnorm)
        @printf("  k_dominant=%d  gap_dominant=%.6e  overlap=%.6f\n",
                k_dominant, gap_dominant, ov_dominant)
        for k in 0:min(K, length(overlaps)-1)
            @printf("    overlap_k%d = %.6f  (E_k - E_0 = %.6e)\n",
                    k, overlaps[k+1], energies[k+1] - E0)
        end

        row = @sprintf("%d,%s,%s,%d,%.10f,%.10f",
                       L, b, z2obc_label, k_dominant, gap_dominant, ov_dominant)
        row *= "," * join([@sprintf("%.10f", o) for o in overlaps], ",")
        push!(rows, row)
    end

    fname = "convergence_overlap.csv"
    open(joinpath(DATA_DIR, fname), "w") do io
        for r in rows; println(io, r); end
    end
    println("\n[overlap] Wrote ", joinpath(DATA_DIR, fname))
    return nothing
end

###############################################################################
# 14b. MOMENTUM-ZERO OVERLAP TARGETING
#
# Replace the center-localized O with the symmetric sum
#   O_p=0 = (1/√n_terms) Σ_n X_{matter n}·σᶻ_{n,n+1}·X_{matter n+1}.
# For a translation-invariant problem this creates a momentum-zero
# meson eigenstate (a single state of definite p=0), so the overlap with
# the spectrum is concentrated rather than spread, and the "lowest k with
# nontrivial overlap" identifies the meson cleanly.
#
# Pick rule: lowest k ≥ 1 with |⟨ψ_k|O_p=0|ψ_0⟩|² ≥ overlap_threshold.
# Also report the argmax over k>0 for comparison.
###############################################################################

function convergence_overlap_pzero(; N_list::Vector{Int}=[4, 6, 8, 10, 12],
                                    K::Int=12,
                                    m::Float64=1.125, η::Float64=1.0,
                                    α::Float64=1.0,
                                    overlap_threshold::Float64=0.01,
                                    nsweeps_gs::Int=80,
                                    nsweeps_ex::Int=80,
                                    maxdim_gs::Int=300,
                                    maxdim_ex::Int=300,
                                    cutoff::Float64=1e-11,
                                    weight::Float64=200.0,
                                    bg_left::Int=+1,
                                    bg_right::Int=+1,
                                    seed::Int=0,
                                    early_stop_tol::Float64=1e-7,
                                    fname_suffix::String="")
    _ensure_data_dir()
    if seed > 0
        Random.seed!(seed)
        @printf("\n[overlap_pzero] seeded with %d\n", seed)
    end
    @printf("\n[overlap_pzero] |P_p=0⟩=O_p=0|Ω⟩ overlap targeting, K=%d, τ=%.4f\n",
            K, overlap_threshold)
    @printf("[overlap_pzero] nsweeps_gs=%d, nsweeps_ex=%d, maxdim_gs=%d, maxdim_ex=%d, cutoff=%.0e, weight=%.1f\n",
            nsweeps_gs, nsweeps_ex, maxdim_gs, maxdim_ex, cutoff, weight)

    cases = [
        (:open_site, :drop),
        (:open_site, :truncate_xz),
        (:PBC,       :drop),
    ]

    rows = String[]
    overlap_cols = join(["overlap_k$(k)" for k in 0:K], ",")
    sigma_cols   = join(["sigma_k$(k)" for k in 0:K], ",")
    gap_cols     = join(["gap_k$(k)" for k in 0:K], ",")
    push!(rows, "L,boundary,z2_obc_boundary,k_meson,gap_meson,overlap_meson,sigma_meson,k_argmax,gap_argmax,overlap_argmax,sigma_argmax,barE_gap,sigma_barE,W_total,n_terms,$(overlap_cols),$(sigma_cols),$(gap_cols)")

    for L in N_list, (b, z2obc) in cases
        z2obc_label = b === :PBC ? "n/a" : String(z2obc)
        @printf("\n[overlap_pzero] L=%d  boundary=%s  z2_obc=%s\n", L, b, z2obc_label)
        _t_h = @elapsed begin
            sites = build_sites(L; boundary=b)
            H     = build_hamiltonian(sites, m, η; boundary=b, gauge_law=:z2, α=α,
                                      z2_obc_boundary=z2obc,
                                      bg_left=bg_left, bg_right=bg_right)
        end

        if b === :PBC
            sweeps_gs     = dmrg_sweeps_pbc(nsweeps=nsweeps_gs, maxdim_final=maxdim_gs, cutoff=cutoff)
            sweeps_higher = dmrg_sweeps_pbc(nsweeps=nsweeps_ex, maxdim_final=maxdim_ex, cutoff=cutoff)
        else
            sweeps_gs     = dmrg_sweeps(nsweeps=nsweeps_gs, maxdim_final=maxdim_gs, cutoff=cutoff)
            sweeps_higher = dmrg_sweeps(nsweeps=nsweeps_ex, maxdim_final=maxdim_ex, cutoff=cutoff)
        end

        local energies, states
        _t_dmrg = @elapsed try
            energies, states = compute_excited_states(sites, H, sweeps_gs;
                                                       k=K, weight=weight, χ_init=10,
                                                       higher_sweeps=sweeps_higher,
                                                       early_stop_tol=early_stop_tol)
        catch err
            @warn "DMRG failed at L=$L boundary=$b z2_obc=$z2obc_label: $err"
            continue
        end

        # Per-state energy variance σ_k = √(⟨ψ_k|H²|ψ_k⟩ − E_k²).
        # For an exact eigenstate this is 0; non-zero σ_k indicates the
        # DMRG truncation error on that eigenvalue.
        _t_var = @elapsed begin
            sigmas = Float64[]
            for (k, ψk) in enumerate(states)
                try
                    # Contract ⟨ψ|H²|ψ⟩ directly without materializing H|ψ⟩.
                    h2 = real(inner(H, ψk, H, ψk))
                    σ2 = h2 - energies[k]^2
                    push!(sigmas, sqrt(max(σ2, 0.0)))
                catch err
                    @warn "variance failed at k=$(k-1): $err"
                    push!(sigmas, NaN)
                end
            end
        end

        E0   = energies[1]
        psi0 = states[1]

        _t_ov = @elapsed begin
            O_mpo, n_terms = build_O_pzero_mpo(sites, L, b)
            overlaps = Float64[]
            for k in 0:K
                kidx = k + 1
                if kidx > length(states)
                    push!(overlaps, NaN)
                else
                    amp = inner(states[kidx]', O_mpo, psi0)
                    push!(overlaps, abs2(amp))
                end
            end
        end
        @printf("  O_p=0 with %d terms\n", n_terms)
        @printf("  [PROFILE] L=%d %s %s: H_build=%.2fs  DMRG=%.2fs  overlap=%.2fs  variance=%.2fs\n",
                L, string(b), string(z2obc), _t_h, _t_dmrg, _t_ov, _t_var)
        flush(stdout)

        # Pick rule 1: lowest k >= 1 with overlap >= threshold.
        k_meson = -1
        for k in 1:K
            kidx = k + 1
            if kidx <= length(overlaps) && isfinite(overlaps[kidx]) &&
               overlaps[kidx] >= overlap_threshold
                k_meson = k
                break
            end
        end
        gap_meson = (k_meson == -1) ? NaN : energies[k_meson + 1] - E0
        ov_meson  = (k_meson == -1) ? NaN : overlaps[k_meson + 1]

        # Pick rule 2: argmax over k=1..K.
        k_argmax = -1; max_ov = -Inf
        for k in 1:K
            kidx = k + 1
            if kidx <= length(overlaps) && isfinite(overlaps[kidx]) &&
               overlaps[kidx] > max_ov
                max_ov   = overlaps[kidx]
                k_argmax = k
            end
        end
        gap_argmax = (k_argmax == -1) ? NaN : energies[k_argmax + 1] - E0
        ov_argmax  = (k_argmax == -1) ? NaN : overlaps[k_argmax + 1]

        # Pull σ_k for the meson and argmax picks (combined variance of the
        # gap = √(σ_meson² + σ_0²), since gap = E_k − E_0).
        σ_gs = sigmas[1]
        σ_meson_combined  = k_meson  == -1 ? NaN : sqrt(sigmas[k_meson + 1]^2 + σ_gs^2)
        σ_argmax_combined = k_argmax == -1 ? NaN : sqrt(sigmas[k_argmax + 1]^2 + σ_gs^2)

        # Per-k gap = E_k - E0, for post-processing.
        gaps = Float64[energies[k+1] - E0 for k in 0:K]

        # Spectral first moment of the gap, weighted by |⟨k|O_p=0|ψ_0⟩|² (k≥1):
        #   ̄gap = Σ_{k≥1} w_k · gap_k / Σ_{k≥1} w_k,   w_k = overlap_k.
        # Error: σ²(̄gap) = Σ_k (w_k/W)² σ_k² + σ_0² (gs uncertainty independent).
        sum_w  = 0.0
        sum_Ew = 0.0
        sum_σ²w² = 0.0
        for k in 1:K
            kidx = k + 1
            if kidx <= length(overlaps) && isfinite(overlaps[kidx])
                w = overlaps[kidx]
                sum_w  += w
                sum_Ew += gaps[kidx] * w
                sum_σ²w² += (kidx <= length(sigmas) && isfinite(sigmas[kidx]) ? sigmas[kidx]^2 : 0.0) * w^2
            end
        end
        barE_gap = sum_w > 0 ? sum_Ew / sum_w : NaN
        σ_barE   = sum_w > 0 ? sqrt(sum_σ²w² / sum_w^2 + σ_gs^2) : NaN
        W_total  = sum_w   # sum of overlap weights (≈ ⟨P|P⟩ if k completeness held)

        # Verify |P⟩ norm (informative — not 1 because O_p=0 isn't unitary).
        # ⟨P|P⟩ = ⟨ψ0|O†O|ψ0⟩ = inner(O, ψ0, O, ψ0) — avoid materializing O|ψ0⟩.
        Pnorm = sqrt(max(real(inner(O_mpo, psi0, O_mpo, psi0)), 0.0))
        @printf("  |P_p=0⟩ norm = %.6f\n", Pnorm)
        @printf("  k_meson=%d (gap=%.4f±%.4f, ov=%.4f)   k_argmax=%d (gap=%.4f±%.4f, ov=%.4f)\n",
                k_meson, gap_meson, σ_meson_combined, ov_meson,
                k_argmax, gap_argmax, σ_argmax_combined, ov_argmax)

        row = @sprintf("%d,%s,%s,%d,%.10f,%.10f,%.10f,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d",
                       L, b, z2obc_label, k_meson, gap_meson, ov_meson, σ_meson_combined,
                       k_argmax, gap_argmax, ov_argmax, σ_argmax_combined,
                       barE_gap, σ_barE, W_total, n_terms)
        row *= "," * join([@sprintf("%.10f", o) for o in overlaps], ",")
        row *= "," * join([@sprintf("%.10f", s) for s in sigmas], ",")
        row *= "," * join([@sprintf("%.10f", g) for g in gaps], ",")
        push!(rows, row)
    end

    fname = "convergence_overlap_pzero$(fname_suffix).csv"
    open(joinpath(DATA_DIR, fname), "w") do io
        for r in rows; println(io, r); end
    end
    println("\n[overlap_pzero] Wrote ", joinpath(DATA_DIR, fname))
    return nothing
end

###############################################################################
# 14c. BOUNDARY-CHARGE SCAN (4 sign combinations at fixed L, OBC :truncate_xz)
###############################################################################

function bg_charge_scan_pzero(; L::Int=8, K::Int=12,
                                m::Float64=1.125, η::Float64=1.0, α::Float64=1.0,
                                nsweeps_gs::Int=20, nsweeps_ex::Int=28,
                                maxdim_gs::Int=400, maxdim_ex::Int=500,
                                cutoff::Float64=1e-11, weight::Float64=100.0,
                                overlap_threshold::Float64=0.01)
    _ensure_data_dir()
    @printf("\n[bg_scan] OBC :truncate_xz, L=%d, scanning (bg_left, bg_right) ∈ {±1}²\n", L)

    sites = build_sites(L; boundary=:open_site)

    rows = String[]
    overlap_cols = join(["overlap_k$(k)" for k in 0:K], ",")
    push!(rows, "L,bg_left,bg_right,E0,E1_minus_E0,E2_minus_E0,k_meson,gap_meson,overlap_meson,k_argmax,gap_argmax,overlap_argmax,$(overlap_cols)")

    for bg_left in (+1, -1), bg_right in (+1, -1)
        @printf("\n[bg_scan] L=%d  bg_left=%+d  bg_right=%+d\n", L, bg_left, bg_right)
        H = build_hamiltonian(sites, m, η; boundary=:open_site, gauge_law=:z2, α=α,
                              z2_obc_boundary=:truncate_xz,
                              bg_left=bg_left, bg_right=bg_right)
        sweeps_gs = dmrg_sweeps(nsweeps=nsweeps_gs, maxdim_final=maxdim_gs, cutoff=cutoff)
        sweeps_ex = dmrg_sweeps(nsweeps=nsweeps_ex, maxdim_final=maxdim_ex, cutoff=cutoff)
        energies, states = compute_excited_states(sites, H, sweeps_gs;
                                                   k=K, weight=weight, χ_init=10,
                                                   higher_sweeps=sweeps_ex)
        E0 = energies[1]
        gap1 = energies[2] - E0
        gap2 = energies[3] - E0

        O_mpo, _ = build_O_pzero_mpo(sites, L, :open_site)
        overlaps = Float64[abs2(inner(states[k+1]', O_mpo, states[1])) for k in 0:K]

        k_meson = -1
        for k in 1:K
            if isfinite(overlaps[k+1]) && overlaps[k+1] >= overlap_threshold
                k_meson = k; break
            end
        end
        gap_meson = (k_meson == -1) ? NaN : energies[k_meson+1] - E0
        ov_meson  = (k_meson == -1) ? NaN : overlaps[k_meson+1]

        k_argmax = -1; max_ov = -Inf
        for k in 1:K
            if isfinite(overlaps[k+1]) && overlaps[k+1] > max_ov
                max_ov = overlaps[k+1]; k_argmax = k
            end
        end
        gap_argmax = (k_argmax == -1) ? NaN : energies[k_argmax+1] - E0
        ov_argmax  = (k_argmax == -1) ? NaN : overlaps[k_argmax+1]

        @printf("  E1-E0=%.4f  E2-E0=%.4f  k_meson=%d (gap=%.4f, ov=%.4f)  k_argmax=%d (gap=%.4f, ov=%.4f)\n",
                gap1, gap2, k_meson, gap_meson, ov_meson, k_argmax, gap_argmax, ov_argmax)

        row = @sprintf("%d,%+d,%+d,%.10f,%.10f,%.10f,%d,%.10f,%.10f,%d,%.10f,%.10f",
                       L, bg_left, bg_right, E0, gap1, gap2,
                       k_meson, gap_meson, ov_meson, k_argmax, gap_argmax, ov_argmax)
        row *= "," * join([@sprintf("%.10f", o) for o in overlaps], ",")
        push!(rows, row)
    end

    fname = @sprintf("bg_charge_scan_L%d.csv", L)
    open(joinpath(DATA_DIR, fname), "w") do io
        for r in rows; println(io, r); end
    end
    println("[bg_scan] Wrote ", joinpath(DATA_DIR, fname))
    return rows
end

###############################################################################
# 15. PRETTY PRINTING
###############################################################################

function print_es_summary(λ::Vector{Float64}, bond::Int; n_print=10)
    println("\n─── Entanglement Spectrum at bond $bond ───")
    println("  Rank: $(length(λ))")
    println("  Top $(min(n_print, length(λ))) Schmidt values:")
    for (i, v) in enumerate(λ[1:min(n_print,end)])
        @printf("    λ_%02d = %.8f   ξ_%02d = %.4f\n", i, v, i, -log(v^2))
    end
    pairs, dr = es_degeneracy_check(λ)
    @printf("  Degenerate pairs: %d  (%.0f%% of levels)\n", pairs, 100dr)
    println()
end

###############################################################################
# 16. MAIN DRIVER — single-point diagnostic example (OBC, paper Eq. 2)
###############################################################################

function main(; N_matter::Int=8, m::Float64=1.125, η::Float64=1.0)
    println("="^60)
    println("  SPT / TOPOLOGICAL PHASE DIAGNOSTICS via DMRG")
    println("="^60)

    sites  = build_sites(N_matter; boundary=:open_site)
    N_tot  = length(sites)
    println("\nSystem: N_matter=$N_matter, N_total=$N_tot (open_site)")
    println("Parameters: m=$m, η=$η  (gauge_law=:z2)\n")

    H      = build_hamiltonian(sites, m, η; boundary=:open_site, gauge_law=:z2)
    sweeps = dmrg_sweeps(maxdim_final=200, cutoff=1e-10)

    println("Running ground state DMRG...")
    psi0_init = randomMPS(sites, 10)
    E0, psi   = dmrg(H, psi0_init, sweeps; outputlevel=0)
    @printf("\nE0 = %.10f  (E0/site = %.6f)\n", E0, E0/N_tot)

    println("\nRunning excited state DMRG...")
    psi1_init = randomMPS(sites, 10)
    E1, psi1  = dmrg(H, [psi], psi1_init, sweeps; weight=20.0, outputlevel=0)
    gap = E1 - E0
    @printf("E1 = %.10f,  Gap = %.10f\n", E1, gap)

    println("\nDiagnostics:")
    d = diagnose(psi, sites; boundary=:open_site, gauge_law=:z2, gap=gap)
    @printf("  S_mid       = %.4f\n", d.S_mid)
    @printf("  deg_ratio   = %.4f\n", d.deg_ratio)
    @printf("  Ostr (ZXZ)  = %.6f\n", d.Ostr_ZX)
    @printf("  Ostr (XZX)  = %.6f\n", d.Ostr_XZ)
    @printf("  S_edge      = %.4f\n", d.S_edge)
    @printf("  phase       = %s\n", d.phase)
    return psi, sites, d
end

###############################################################################
# Script entrypoint:  only runs when the file is executed directly.
###############################################################################

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n############  Section III smoke run (higgs_obc_boundary=:drop)  ############")
    section_iii_smoke_run(; higgs_obc_boundary=:drop)
    println("\n############  Section III smoke run (higgs_obc_boundary=:truncate_xx)  ############")
    section_iii_smoke_run(; higgs_obc_boundary=:truncate_xx)
    println("\n############  Convergence run (z2_obc_boundary=:drop)  ############")
    convergence_run(; N_list=[4, 6, 8, 10, 12], k_max=4,
                      z2_obc_boundary=:drop)
    println("\n############  Convergence run (z2_obc_boundary=:truncate_xz)  ############")
    convergence_run(; N_list=[4, 6, 8, 10, 12], k_max=4,
                      z2_obc_boundary=:truncate_xz)
    println("\n############  Phase scan smoke (z2, open_site, L=4)  ############")
    phase_scan_smoke(; N_matter=4, boundary=:open_site, gauge_law=:z2)
    println("\n############  Phase scan smoke (higgs_spt, open_site, L=4)  ############")
    phase_scan_smoke(; N_matter=4, boundary=:open_site, gauge_law=:higgs_spt)
    println("\n############  Sector-filtered meson gap (Option 1, k_max=12)  ############")
    convergence_sector_filtered()
    println("\n############  Overlap-targeted meson (Option 2, K=8)  ############")
    convergence_overlap()
    println("\n############  Momentum-zero overlap meson (K=12)  ############")
    convergence_overlap_pzero(; K=12, overlap_threshold=0.01)
end
