
function mpo_to_array(H0, sites)
    h0 = prod(H0)
    C1 = combiner(sites)
    c = combinedind(C1)
    hmat = Matrix(C1 * h0 * C1', c, c')
    return hmat
end
