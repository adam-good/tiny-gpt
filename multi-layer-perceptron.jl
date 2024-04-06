using Flux
using Flux.Data: DataLoader
using Flux: onecold, logitcrossentropy, throttle, params
using CUDA
using MLDatasets

if has_cuda()
    @info "CUDA: On"
    CUDA.allowscalar(false)
end

Base.@kwdef mutable struct Args
    learning_rate::Float64 = 3e-4
    batchsize::Int = 1024
    epochs::Int = 10
    device::Function = cpu
end

function get_mnist(args)

    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    xtrain, ytrain = MLDatasets.MNIST(split=:train)[:]#.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST(split=:test)[:]#.testdata(Float32)

    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    ytrain = Flux.onehotbatch(ytrain, 0:9)
    ytest = Flux.onehotbatch(ytest, 0:9)

    train_data = DataLoader( (xtrain, ytrain), batchsize=args.batchsize )
    test_data = DataLoader( (xtest, ytest), batchsize=args.batchsize )

    @info "Dataset: MNIST"
    (train_data, test_data)

end

function build_model(; input_size=(28,28,1), output_size=10)
    Chain(
        Dense(prod(input_size), 32, relu),
        Dense(32, output_size)
    )
end

function loss_all(dataloader, model)
    loss = 0f0
    for (x,y) in dataloader
        loss += logitcrossentropy(model(x), y)
    end
    loss / length(dataloader)
end

function accuracy(dataloader, model)
    acc = 0
    for (x,y) in dataloader
        # x̂ = onecold(cpu(model(x)))
        # y = onecold(cpu(y))*1
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc / length(dataloader)
end

function train!(model, data, args; kws...)
    @info "Training Model"
    @info "Learning Rate: $(args.learning_rate)"
    @info "Epochs: $(args.epochs)"
    @info "Device: $(args.device)"
    train_data, test_data = data

    train_data = args.device.(train_data)
    test_data = args.device.(test_data)
    model = args.device(model)

    loss(x,y) = logitcrossentropy(model(x), y)

    opt = Adam(args.learning_rate)

    eval_callback = () -> @show(loss_all(train_data, model))

    for epoch in 1:args.epochs
        @info "Epoch: $epoch"
        Flux.train!(loss, params(model), train_data, opt)
        eval_callback()
    end

    @show accuracy(train_data, model)
    @show accuracy(test_data, model)
end

args = Args()
data = get_mnist(args)
model = build_model()
train!(model, data, args)