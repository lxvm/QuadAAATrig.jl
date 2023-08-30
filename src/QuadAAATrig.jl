"""
This package provides the routine [`quadaaatrig`](@ref) for integration of
periodic, multi-dimensional, scalar functions via rational approximation with
the AAA algorithm provided by `BaryRational`.

TODO: rewrite AAA convergence loop so that it also checks the convergence of the estimated
integral
"""
module QuadAAATrig

using Printf

export quadaaatrig, QuadAAATrigJL, aaa, continuumquadaaatrig, ContinuumQuadAAATrigJL, aaaz

include("BaryRational.jl")
using .BaryRational

include("ContinuumAAA.jl")
using .ContinuumAAA

velocity(a::BaryRational.AAAapprox, z::T) where {T} =
    velocity(z, a.f, a.x, a.w)
function velocity(z::T, f::Vector{T}, x::Vector{T}, w::Vector{T}) where {T}
    num = zero(T)
    den = zero(T)
    @inbounds for j in eachindex(f)
        idz = inv(z - x[j])
        t = w[j] * f[j] * idz
        num += t
        den += t * idz
    end
    fz = num / den
    fz = isfinite(fz) ? fz : throw(DomainError("Found $fz at $z"))
end

struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x, p) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Y
    x::X
    max_batch::Int # maximum number of x to supply in parallel
    function BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch::Integer=typemax(Int))
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{typeof(f!),typeof(y),typeof(x)}(f!, y, x, max_batch)
    end
end


"""
    BatchIntegrand(f!, y, x; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` with pre-allocated buffers.
"""
BatchIntegrand(f!, y, x; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, y, x, max_batch)

"""
    BatchIntegrand(f!, y::Type, x::Type=Nothing; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` whose range type is known. The domain type is optional.
Array buffers for those types are allocated internally.
"""
BatchIntegrand(f!, Y::Type, X::Type=Nothing; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, Y[], X[], max_batch)


function quadaaatrig(f, a::T, b::T, npt; atol=nothing, rtol=nothing, fbuffer = nothing, verb = false, kws...) where {T}

    X = range(a, b, length=npt+1)[1:npt]

    F = if f isa BatchIntegrand
        if eltype(f.x) == Nothing
            f.f!(f.y, X)
        else
            f.f!(f.y, f.x .= X)
        end
        f.y
    elseif fbuffer === nothing
        f.(X)
    else
        fbuffer .= f.(X)
    end

    Iptr = sum(F)*(b-a)/npt # use the ptr to obtain an order of magnitude integral estimate
    abstol = atol === nothing ? zero(abs(Iptr)) : atol
    reltol = rtol === nothing ? (iszero(abstol) ? sqrt(sqrt(eps(one(abstol))))^3 : zero(one(abstol))) : rtol
    # aaa tol is pointwise, however quad tol is global, and by residue theorem ∫f = 2πim*Res
    # means that the pointwise aaa tol on f (hence Res(f))
    twopi = pi*(one(T) + one(T))
    tol = max(abstol, reltol*abs(Iptr))/(b-a) # rescale pointwise error to global
    Iaaa, npol = discresiaaa(F, range(zero(one(T)), twopi, length=npt+1)[1:npt]; tol=tol, verb=verb, kws...)
    return (Iaaa*(b-a)/twopi, npol)
end
quadaaatrig(f, a, b, npt; kws...) = quadaaatrig(f, promote(a,b)..., npt; kws...)

"""
    discresiaaa(F, X; verb=0, UCdist=0)

Integrates a 2π-periodic function's samples `F` at nodes `X` by constructing a
AAA approximant and summing its residues in the unit disc. The keyword `UCdist`
sets a tolerance for annulus about the unit circle where poles are included or
excluded based on their velocity.
"""
function discresiaaa(F::AbstractVector{<:Number}, X::AbstractVector{<:Real};
    tol=sqrt(sqrt(eps(eltype)))^3, mmax=100, verb=true, clean=true, UCdist=nothing, zbuffer=nothing)
    Z = zbuffer === nothing ? exp.(im.*X) : (resize!(zbuffer, length(X)) .= exp.(im.*X))
    a = aaa(Z, F ./ (im .* Z); tol=tol, verbose=verb, mmax=mmax, clean=clean) # integrand*jacobian : z = exp(im*x); dz = im*z*dx
    pol, res, _ = prz(a)            # poles, residues, and zeros
    return discresi(pol, res, UCdist === nothing ? abs(zero(eltype(pol))) : UCdist, verb)
end
function discresi(pol, res, UCdist, verb=true)
    A = zero(complex(float(eltype(res))))
    for (z,r) in zip(pol,res)
        verb && @printf "\tpole |z|=%.15f ang=%.6f: " abs(z) angle(z)
        if abs(z)>one(UCdist)+UCdist           # outside
            verb && @printf "\texclude\n"
        elseif abs(z)<=one(UCdist)-UCdist      # inside, include
            A += π*im*(r + r)               # the residue thm
            verb && @printf "\tres=%g+%gi\n" real(r) imag(r)
        else                           # handle on (eps-close to) unit circ
            # estimate velocity (h′~f′/f^2) of simple pole (f(x)=1/h(x)) to
            # decide if pole is approaching from outside disc
            # da = -deriv(a, z)/a(z)^2    # probably want to stably compute this (the derivative may not exist)
            da = velocity(a, z)        # probably want to stably compute this (the derivative may not exist)
            if real(da)>0.0            # pole approach from outside
                verb && @printf "UC\texclude\tda=%g+i%g\n" real(da) imag(da)
            else                       # include
                A += π*im*(r + r)
                verb && @printf "UC\tres=%g+%gi\n" real(r) imag(r)
            end
        end
    end
    return A, length(pol)
end

function QuadAAATrigJL(args...; kws...)
    ext = Base.get_extension(QuadAAATrig, :AutoBZCoreExt)
    if ext !== nothing
        return ext.QuadAAATrigAlg(args...; kws...)
    else
        error("AutoBZCore extension not loaded")
    end
end

# TODO add a local pole subtraction method based on AAA for use on non-periodic functions

# This is a continuumAAA based method

function continuumquadaaatrig(f, a::T, b::T; atol, verb = false, kws...) where {T}

    # aaa tol is pointwise, however quad tol is global, and by residue theorem ∫f = 2πim*Res
    # means that the pointwise aaa tol on f (hence Res(f))
    twopi = pi*(one(T) + one(T))
    s = (b-a)/twopi

    # we need to pick a different branch from the standard logarithm
    g = z -> f((-real(im*log(-z))+pi)*s) / (im*z)

    Iaaa, npol = discresicontinuumaaa(g; verb = verb, tol=atol/(b-a), numtype=typeof(twopi), kws...)
    return (Iaaa*s, npol)
end
continuumquadaaatrig(f, a, b; kws...) = continuumquadaaatrig(f, promote(a,b)...; kws...)

function discresicontinuumaaa(f::F; verb=true, UCdist=nothing, kws...) where {F}
    _, pol, res, = aaaz(f; kws...)
    return discresi(pol, res, UCdist === nothing ? abs(zero(eltype(pol))) : UCdist, verb)
end

function ContinuumQuadAAATrigJL(args...; kws...)
    ext = Base.get_extension(QuadAAATrig, :AutoBZCoreExt)
    if ext !== nothing
        return ext.ContinuumQuadAAATrigAlg(args...; kws...)
    else
        error("AutoBZCore extension not loaded")
    end
end

end # module QuadAAATrig
