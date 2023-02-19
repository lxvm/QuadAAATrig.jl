"""
This package provides the routine [`quadaaatrig`](@ref) for integration of
periodic, multi-dimensional, scalar functions via rational approximation with
the AAA algorithm provided by `BaryRational`.
"""
module QuadAAATrig

using Printf

using BaryRational
using FourierSeriesEvaluators

export quadaaatrig

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

quadaaatrig(f, s::FourierSeries, params; verb=0, density=2, UCdist=1e-13) =
    doquadaaa(Val(ndims(s)), f, s, params, period(s), density, UCdist, verb)

function doquadaaa(::Val{1}, f, s, params, period, density, UCdist, verb)
    npt = round(Int, density*size(s.c,1))
    X = range(0, period[1], length=npt+1)[1:npt]
    F = map(x -> f(s(x), params), X)
    discresiaaa(F, X; verb=verb, UCdist=UCdist)
end

function doquadaaa(::Val{N}, f, s, params, period, density, UCdist, verb) where N
    npt = round(Int, density*size(s.c,N))
    X = range(0, period[N], length=npt)[1:npt]
    F = map(x -> doquadaaa(Val(N-1), f, contract(s, x, Val(N)), params, period, density, UCdist, verb), X)
    discresiaaa(F, X; verb=verb, UCdist=UCdist)
end

"""
    discresiaaa(F, X; verb=0, UCdist=0)

Integrates a 2π-periodic function's samples `F` at nodes `X` by constructing a
AAA approximant and summing its residues in the unit disc. The keyword `UCdist`
sets a tolerance for annulus about the unit circle where poles are included or
excluded based on their velocity.
"""
function discresiaaa(F, X; verb=0, UCdist=0.0)
    Z = exp.(im.*X)
    a = aaa(Z, F ./ (im .* Z)) # integrand*jacobian : z = exp(im*x); dz = im*z*dx
    pol, res, _ = prz(a)            # poles, residues, and zeros
    A = zero(complex(float(eltype(res))))
    for (z,r) in zip(pol,res)
        verb==0 || @printf "\tpole |z|=%.15f ang=%.6f: " abs(z) angle(z)
        if abs(z)>1.0+UCdist           # outside
            verb==0 || @printf "\texclude\n"
        elseif abs(z)<=1.0-UCdist      # inside, include
            A += 2π*im*r               # the residue thm
            verb==0 || @printf "\tres=%g+%gi\n" real(r) imag(r)
        else                           # handle on (eps-close to) unit circ
            # estimate velocity (h′~f′/f^2) of simple pole (f(x)=1/h(x)) to
            # decide if pole is approaching from outside disc
            # da = -deriv(a, z)/a(z)^2    # probably want to stably compute this (the derivative may not exist)
            da = velocity(a, z)        # probably want to stably compute this (the derivative may not exist)
            if real(da)>0.0            # pole approach from outside
                verb==0 || @printf "UC\texclude\tda=%g+i%g\n" real(da) imag(da)
            else                       # include
                A += 2π*im*r
                verb==0 || @printf "UC\tres=%g+%gi\n" real(r) imag(r)
            end
        end
    end
    A
end


end # module QuadAAATrig
