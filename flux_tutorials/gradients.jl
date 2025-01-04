using Flux
using Flux: gradient

IN::UInt = 10
H_SIZE::UInt = 5
OUT::UInt = 2

struct Layer
    W::Matrix{Float64}
    b::Vector{Float64}
end

function Layer(in::UInt, out::UInt)
    Layer(
        rand(out, in),
        rand(out)
    )
end

L1 = Layer(IN, H_SIZE)
L2 = Layer(H_SIZE, OUT)

# W = rand(OUT, IN)
# b = rand(OUT)

predict(x) = L1.W * x .+ L1.b

function loss(x, y)
    ŷ = predict(x)
    sum((ŷ .- y) .^ 2)
end

x, y = rand(IN), rand(OUT)
L = loss(x, y)
@info "Loss: $L"

@info "Performing Gradient Descent"
gs = gradient(() -> loss(x, y), Flux.params(L1.W, L1.b))

# α = 0.1
# W̄ = gs[L1.W]
# L1.W .-= α .* W̄

# L = loss(x, y)
# @info "Loss: $L"
