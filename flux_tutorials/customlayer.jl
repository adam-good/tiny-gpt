using Flux: @layer
using Flux

struct DenseLayer
    chain::Chain
    in::UInt
    out::UInt
    Weight::Matrix{Float64}
    bias::Vector{Float64}
    activation::Function
end

function DenseLayer(in::UInt, out::UInt; activation::Function=(x) -> x)
    DenseLayer(
        Chain(),
        in, out,
        rand(out, in), rand(out),
        activation
    )
end

"""
    (l::DenseLayer)(x)

TBW
"""
function (l::DenseLayer)(x)
    ŷ = l.activation(l.Weight * x + l.bias)
    return l.chain(ŷ)
end

@layer DenseLayer
