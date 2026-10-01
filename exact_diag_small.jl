###############################################################################
# Exact diagonalisation at N_s = 4, 6 for the paper's Sec. III parameter point
# (m0 = 0.1, eta = 0.5).
#
# For PBC and for OBC with each of the four boundary-charge assignments
# (q_L, q_R) it writes the meson-band moments of the volume-averaged operator,
#     W = sum_n w_n,   Dbar = sum_n w_n Delta_n / W     (Delta_n < 2),
# to data/exact_diag/exact_moments.csv and prints the low-lying spectrum.
#
#   julia --project=. exact_diag_small.jl
###############################################################################

using ITensors, ITensorMPS, LinearAlgebra, Printf

include(joinpath(@__DIR__, "z2-gauge-theory.jl"))

const DELTA_CUT = 2.0

function exact_spectrum(L, boundary, z2obc, qL, qR; m0=0.1, eta=0.5)
    p = Data.Params(; m0=m0, eta=eta, alpha=1.0, Lphys=L, L=L)
    H, sites = build_H(p; boundary=boundary, gauge_law=:z2,
                       z2_obc_boundary=z2obc, bg_left=qL, bg_right=qR)
    F = eigen(Hermitian(mpo_to_array(H, sites)))
    O = mpo_to_array(make_O_pzero_mpo(sites, L, boundary), sites)
    weights = abs2.(F.vectors' * (O * F.vectors[:, 1]))
    return F.values, weights
end

outdir = joinpath(@__DIR__, "data", "exact_diag")
mkpath(outdir)
open(joinpath(outdir, "exact_moments.csv"), "w") do io
    println(io, "L,boundary,q_left,q_right,E0,n_band,W,Dbar,gap_min")
    for L in (4, 6)
        configs = [(:PBC, :drop, -1, -1), (:open_site, :truncate_xz, -1, -1),
                   (:open_site, :truncate_xz, 1, -1), (:open_site, :truncate_xz, -1, 1),
                   (:open_site, :truncate_xz, 1, 1)]
        for (boundary, z2obc, qL, qR) in configs
            E, w = exact_spectrum(L, boundary, z2obc, qL, qR)
            gaps = E .- E[1]
            band = [n for n in 2:length(gaps) if gaps[n] < DELTA_CUT]
            W = sum(w[band])
            Dbar = sum(w[band] .* gaps[band]) / W
            @printf(io, "%d,%s,%d,%d,%.12f,%d,%.12f,%.12f,%.12f\n",
                    L, boundary, qL, qR, E[1], length(band), W, Dbar, gaps[2])
            @printf("L=%d %-9s q=(%+d,%+d) E0=%.6f  n_band=%d  W=%.10f  Dbar=%.10f\n",
                    L, boundary, qL, qR, E[1], length(band), W, Dbar)
            println("   Delta[w]: ", join([@sprintf("%.4f[%.3f]", gaps[n], w[n])
                                           for n in 2:min(14, length(gaps))], " "))
        end
    end
end
