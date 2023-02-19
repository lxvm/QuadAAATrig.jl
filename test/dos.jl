using LinearAlgebra

using Plots
using StaticArrays
using OffsetArrays

using QuadAAATrig
using FourierSeriesEvaluators

gloc(h, (ω, η)) = tr(inv(complex(ω,η)*I-h))

function integer_lattice(n=1)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = 1/2n
    end
    FourierSeries(C)
end

function graphene_tb()
    hm = OffsetMatrix(zeros(SMatrix{2,2,ComplexF64,4}, (5,5)), -2:2, -2:2)
    hm[1,1]   = hm[1,-2] = hm[-2,1] = [0 1; 0 0]
    hm[-1,-1] = hm[-1,2] = hm[2,-1] = [0 0; 1 0]
    FourierSeries(hm)
end

function showdos(h, ωs, η; kwargs...)
    glocs = map(ω -> quadaaatrig(gloc, h, (ω, η); kwargs...), ωs)
    plot(ωs, -imag.(glocs) ./ π)
    # plot error for integer_lattice(1)
    # plot(ωs, ω -> (abs(ω)<1 ? 2/sqrt(1-ω^2) : 0.0) + imag(quadaaatrig(gloc, h, (ω, η); kwargs...))/π)
end