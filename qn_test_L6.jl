###############################################################################
# QN-conservation test: matter parity ∏Z_matter is exactly conserved by H.
# Build sites with Z2 QN on matter sites, trivial QN on link sites, and
# verify the result matches non-QN DMRG / exact diagonalization at L=6.
###############################################################################

using ITensors, ITensorMPS, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

function matter_predicate(boundary::Symbol)
    # In :open_site layout, matter sits at odd indices 2j-1 (j=1..L).
    # In :PBC layout, matter sits at even indices 2j (j=1..L).
    if boundary == :open_site
        return i -> isodd(i)
    elseif boundary == :PBC
        return i -> iseven(i)
    else
        error("unsupported boundary $boundary")
    end
end

function make_qn_sites_z2(L::Int, boundary::Symbol)
    N = num_qubits(L, boundary)
    is_m = matter_predicate(boundary)
    sites = Index[]
    for i in 1:N
        if is_m(i)
            # Matter site: 2D, two charge sectors (Z=+1 → P=0, Z=−1 → P=1).
            push!(sites, Index([QN("P", 0, 2) => 1, QN("P", 1, 2) => 1];
                               tags="S=1/2,Site,n=$i"))
        else
            # Link site: 2D, single charge-0 block (Z2 charge always 0).
            push!(sites, Index([QN("P", 0, 2) => 2]; tags="S=1/2,Site,n=$i"))
        end
    end
    return sites
end

function build_H_qn(L::Int, boundary::Symbol, z2_obc::Symbol;
                   m0=0.1, eta=0.5, alpha=1.0, lambda_gauss=20.0,
                   bg_left=-1, bg_right=-1)
    sites = make_qn_sites_z2(L, boundary)

    # Build the same OpSum the non-QN code does. We reuse the SAME helper
    # by calling make_z2_gauge_theory_hamiltonian-style assembly. The OpSum
    # itself doesn't care about QN; the conversion to MPO does.
    # To stay simple, we build a fresh OpSum here, mirroring opsum_data_z2.
    os = OpSum()
    N = length(sites)

    # gauge-law penalty XZX (interior) + boundary XZ/ZX terms if :truncate_xz
    # plus σˣ on links, mass on matter, hopping XZX + YZY.
    # Use the boundary-specific layout maps.
    is_m = matter_predicate(boundary)
    matter_site = j -> begin
        # j-th matter site (j=1..L) -> chain index
        boundary == :open_site ? 2j - 1 : 2j
    end
    link_site = j -> begin
        # link between matter j and matter j+1
        boundary == :open_site ? 2j : 2j + 1
    end

    # σˣ on each gauge link  (electric kinetic)
    if boundary == :open_site
        for j in 1:(L-1)
            os += alpha/2, "X", link_site(j)
        end
    elseif boundary == :PBC
        # links at 1, 3, 5, ..., 2L-1
        for j in 1:L
            kk = 2j - 1 == 0 ? 1 : 2j - 1  # safety
            os += alpha/2, "X", kk
        end
    end

    # Staggered mass on each matter
    for j in 1:L
        sgn = j % 2 == 0 ? 1 : -1
        os += -m0/2 * sgn, "Z", matter_site(j)
    end

    # Hopping XZX + YZY
    if boundary == :open_site
        for j in 1:(L-1)
            mi = matter_site(j); bi = link_site(j); mj = matter_site(j+1)
            os += eta/4, "X", mi, "Z", bi, "X", mj
            os += eta/4, "Y", mi, "Z", bi, "Y", mj
        end
    elseif boundary == :PBC
        for j in 1:L
            mi = matter_site(j); bi = link_site(j); mj = matter_site(j == L ? 1 : j+1)
            os += eta/4, "X", mi, "Z", bi, "X", mj
            os += eta/4, "Y", mi, "Z", bi, "Y", mj
        end
    end

    # Gauss-law soft penalty: -(λ/2)·η_j·X·Z·X (interior matter)
    if boundary == :open_site
        for j in 2:(L-1)
            mi = matter_site(j); l_left = link_site(j-1); l_right = link_site(j)
            ηj = (-1)^j
            os += -lambda_gauss/2 * ηj, "X", l_left, "Z", mi, "X", l_right
        end
        if z2_obc == :truncate_xz
            # boundary_left at j=1: -(λ/2)·η_1·bg_left · Z·X on (matter_1, link_1)
            os += -lambda_gauss/2 * (-1) * bg_left, "Z", matter_site(1), "X", link_site(1)
            # boundary_right at j=L: -(λ/2)·η_L·bg_right · X·Z on (link_{L-1}, matter_L)
            os += -lambda_gauss/2 * (-1)^L * bg_right, "X", link_site(L-1), "Z", matter_site(L)
        end
    elseif boundary == :PBC
        for j in 1:L
            mi = matter_site(j)
            l_left = link_site(j == 1 ? L : j-1)
            l_right = link_site(j)
            ηj = (-1)^j
            os += -lambda_gauss/2 * ηj, "X", l_left, "Z", mi, "X", l_right
        end
    end

    return MPO(os, sites), sites
end

# Test: build the QN-aware H at L=6 OBC and try DMRG with k_max=10.
println("=== L=6 OBC :truncate_xz, bg=(-1,-1)  with QN matter parity ===")
H, sites = build_H_qn(6, :open_site, :truncate_xz)
@printf("Sites:\n")
for (i, s) in enumerate(sites)
    @printf("  site %d: dim=%d, blocks=%s\n", i, dim(s), space(s))
end

# Initial MPS: pick a definite parity sector.
# For our L=6 with -m₀/2·(-1)^n Z_n mass, the ground state should be in the
# matter-parity sector matching the staggered configuration.
# Empirically: ⟨0|∏Z_matter|0⟩ ≈ +1 for L=4, -1 for L=6 (alternates with L).
# Let's start in P=1 (i.e. ⟨∏Z⟩=-1 since (-1)^1=-1 in our convention).
target_state = ["Up" for _ in 1:length(sites)]
# Matter sites at chain indices 1, 3, 5, ..., 2L-1. Set even-j matter to "Dn"
# to get a staggered config in the right parity sector.
for j in 1:6
    msite = 2j - 1
    if j % 2 == 0   # set even matter sites to "Dn"
        target_state[msite] = "Dn"
    end
end
@printf("\nInitial MPS state assignment:\n  %s\n", target_state)
try
    psi0 = MPS(sites, target_state)
    @printf("  initial MPS QN flux = %s\n", flux(psi0))

    sw = Sweeps(80)
    setmaxdim!(sw, 10, 20, 40, 80, 100, 200, 300, 300, 300, 300)
    setcutoff!(sw, 1e-11)
    setnoise!(sw, 1e-7, 1e-8, 1e-9, 0.0)

    tstart = time()
    e0, ψ0 = dmrg(H, psi0, sw; outputlevel=0)
    @printf("\nGround state energy (QN DMRG): %.8f  [%.2fs]\n", e0, time()-tstart)

    # Compute the first 10 excited states in the SAME matter-parity sector.
    energies = [e0]
    psis = [ψ0]
    weight = 200.0
    for k in 1:10
        # random_mps that lives in the same QN sector as psi0
        psi_init = random_mps(sites, target_state; linkdims=2)
        t0 = time()
        ek, ψk = dmrg(H, copy.(psis), psi_init, sw; weight=weight, outputlevel=0)
        @printf("  k=%-2d  E=%.8f  gap=%.6f  [%.2fs]\n", k, ek, ek-e0, time()-t0)
        push!(energies, ek)
        push!(psis, ψk)
    end
    @printf("\nTotal QN-DMRG time (k=0..10): %.2fs\n", time()-tstart)

    # Build O_p=0 MPO + compute overlaps + ̄gap
    O_mpo = make_O_pzero_mpo(sites, 6, :open_site)
    overlaps = Float64[abs2(inner(psis[k+1]', O_mpo, psis[1])) for k in 0:10]
    @printf("\nOverlaps |<k|O|0>|² for k=0..10:\n")
    for k in 0:10
        @printf("  k=%-2d  gap=%.5f  |<k|O|0>|² = %.6f\n", k, energies[k+1]-e0, overlaps[k+1])
    end

    sum_w = sum(overlaps[2:end])
    barE  = sum(overlaps[k+1] * (energies[k+1]-e0) for k in 1:10) / sum_w
    @printf("\nW_total (k≥1) = %.6f\n", sum_w)
    @printf("̄gap = %.6f  (L=6 OBC exact: 1.55357)\n", barE)
catch e
    @printf("ERROR: %s\n", sprint(showerror, e))
    rethrow(e)
end
