using Flux
using Flux.Data: DataLoader
using Flux: train!, onehotbatch, onecold, logitcrossentropy, throttle, params
using Flux: cpu, gpu
using Flux: Chain, Dense, Adam, setup
using CUDA
using CUDA: has_cuda, allowscalar
using MLDatasets
using MLDatasets.MLUtils
using MLDatasets.MLUtils: flatten
using ProgressMeter

if CUDA.has_cuda()
    @info "CUDA: On"
    CUDA.allowscalar(false)
else
    @info "CUDA: Off"
end

Base.@kwdef mutable struct TrainingArgs
    learning_rate::Float64 = 3e-4
    batchsize::Int = 1024
    epochs::Int = 10
    device::Function = cpu
end

function build_feedforward(layer_sizes::Vector{Int}, activation_fns::Vector{Function})
    num_layers = length(layer_sizes) - 1
    if num_layers != length(activation_fns)
        exit(1)
    end
    layer_params = zip(
        layer_sizes[1:num_layers],
        layer_sizes[2:num_layers+1],
        activation_fns
    )
    Chain(
        [Dense(x, y, fn) for (x,y,fn) in layer_params]...
    )
end

function get_mnist(args)

    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    xtrain, ytrain = MLDatasets.MNIST(split=:train)[:]#.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST(split=:test)[:]#.testdata(Float32)

    xtrain = MLUtils.flatten(xtrain)
    xtest = MLUtils.flatten(xtest)

    ytrain = Flux.onehotbatch(ytrain, 0:9)
    ytest = Flux.onehotbatch(ytest, 0:9)

    train_data = DataLoader( (xtrain, ytrain), batchsize=args.batchsize )
    test_data = DataLoader( (xtest, ytest), batchsize=args.batchsize )

    @info "Dataset: MNIST"
    (train_data, test_data)

end

function accuracy(dataloader, model)
    acc = 0
    for (x,y) in dataloader
        ŷ = onecold(model(x) |> cpu)
        y = onecold(y |> cpu)
        acc += sum(ŷ .== y) / size(x, 2)
    end
    acc / length(dataloader)
end

function train_model!(model, data, args; kws...)
    @info "Training Model"
    @info "Learning Rate: $(args.learning_rate)"
    @info "Epochs: $(args.epochs)"
    @info "Device: $(args.device)"
    train_data, test_data = data

    train_data = train_data .|> args.device
    test_data = test_data .|> args.device
    model = model |> args.device

    loss(x,y) = logitcrossentropy(model(x), y)

    opt = Flux.setup(Adam(args.learning_rate), model)

    @showprogress for epoch in 1:args.epochs
        # @info "Epoch: $epoch"
        Flux.train!(model, train_data, opt) do m, x, y
            ŷ = m(x)
            Flux.logitcrossentropy(ŷ, y)
        end
    end

    @show accuracy(train_data, model)
    @show accuracy(test_data, model)
end


img_shape = (28,28,1)
num_classes = 10
layer_sizes = [prod(img_shape), 128, 32, num_classes]
activation_functions = [sigmoid, sigmoid, identity]
model = build_feedforward(layer_sizes, activation_functions)

@info "Model: "
display(model)

args = TrainingArgs()
data = get_mnist(args)
train_model!(model, data, args)