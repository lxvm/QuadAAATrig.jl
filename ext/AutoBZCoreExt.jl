module AutoBZCoreExt

using AutoBZCore
using QuadAAATrig

using AutoBZCore: IntegralAlgorithm, IntegralSolution, endpoints
import AutoBZCore: init_cacheval, do_solve

struct QuadAAATrigAlg{F} <: IntegralAlgorithm
    UCdist::F
    npt::Int
    clean::Bool
end
QuadAAATrigAlg(; UCdist=nothing, npt=50, clean=true) = QuadAAATrigAlg(UCdist, npt, clean)

function init_cacheval(f, dom, p, alg::QuadAAATrigAlg)
    f isa NestedBatchIntegrand && throw(ArgumentError("QuadAAATrigAlg doesn't support nested batching"))
    f isa InplaceIntegrand && throw(ArgumentError("QuadAAATrigAlg doesn't support inplace integrands"))
    return (fbuffer=nothing, zbuffer=Complex{typeof(one(float(real(one(eltype(dom))))))}[])
end

function do_solve(f::F, dom, p, alg::QuadAAATrigAlg, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = 100) where {F}
    I, npoles = quadaaatrig(x -> f(x, p), endpoints(dom)..., alg.npt; atol=abstol, rtol=reltol, mmax=maxiters, UCdist=alg.UCdist,
    clean=alg.clean, fbuffer=cacheval.fbuffer, zbuffer=cacheval.zbuffer)
    return IntegralSolution(I, npoles, true, -1)
end

struct ContinuumQuadAAATrigAlg{F} <: IntegralAlgorithm
    UCdist::F
    lawson::Int
    mero::Bool
end
ContinuumQuadAAATrigAlg(; UCdist=nothing, lawson=0, mero=true) = ContinuumQuadAAATrigAlg(UCdist, lawson, mero)


function init_cacheval(f, dom, p, ::ContinuumQuadAAATrigAlg)
    f isa NestedBatchIntegrand && throw(ArgumentError("ContinuumQuadAAATrigAlg doesn't support nested batching"))
    f isa InplaceIntegrand && throw(ArgumentError("ContinuumQuadAAATrigAlg doesn't support inplace integrands"))
    f isa BatchIntegrand && throw(ArgumentError("ContinuumQuadAAATrigAlg doesn't support batched integrands"))
    return nothing
end

function do_solve(f::F, dom, p, alg::ContinuumQuadAAATrigAlg, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = 150) where {F}
    abstol === nothing && throw(ArgumentError("abstol is a required argument for ContinuumQuadAAATrig"))
    I, npoles = continuumquadaaatrig(x -> f(x, p), endpoints(dom)...; atol=abstol, degree=maxiters, UCdist=alg.UCdist,
        lawson=alg.lawson, mero=alg.mero)
    return IntegralSolution(I, npoles, true, -1)
end

end
