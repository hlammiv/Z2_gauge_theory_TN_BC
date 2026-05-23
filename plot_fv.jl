using CSV, DataFrames, CairoMakie

# Read tight-DMRG CSV with σ.
csv_path = "/home/hlamm/Desktop/QC/circuit_knitting/data/z2_gauge_theory/convergence_overlap_pzero_bg-1-1_m0p1_eta0p5_tight.csv"
df = CSV.read(csv_path, DataFrame)

# Filter to OBC :truncate_xz and PBC.
obc = filter(r -> r.boundary == :open_site && r.z2_obc_boundary == :truncate_xz, df)
pbc = filter(r -> r.boundary == :PBC, df)

sort!(obc, :L)
sort!(pbc, :L)

fig = Figure(; size = (700, 500))
ax = Axis(fig[1,1];
          xlabel = "L (matter sites)",
          ylabel = "M  (gap_argmax with O_p=0)",
          title  = "Finite-volume convergence at (m₀=0.1, η=0.5, α=1, bg=−1,−1)\n" *
                   "z2-gauge-theory.jl, tight DMRG (nsweeps=50, weight=200)")

errorbars!(ax, obc.L, obc.gap_argmax, obc.sigma_argmax;
           color = :blue, whiskerwidth = 10)
scatter!(ax, obc.L, obc.gap_argmax; color = :blue, marker = :circle,
         markersize = 12, label = "OBC :truncate_xz")

errorbars!(ax, pbc.L, pbc.gap_argmax, pbc.sigma_argmax;
           color = :red, whiskerwidth = 10)
scatter!(ax, pbc.L, pbc.gap_argmax; color = :red, marker = :rect,
         markersize = 12, label = "PBC")

axislegend(ax; position = :rb)

# Save both PDF and PNG
save("/home/hlamm/Desktop/QC/circuit_knitting/data/fv_convergence_m0p1_eta0p5.pdf", fig)
save("/home/hlamm/Desktop/QC/circuit_knitting/data/fv_convergence_m0p1_eta0p5.png", fig)
println("Saved plot to data/fv_convergence_m0p1_eta0p5.{pdf,png}")
