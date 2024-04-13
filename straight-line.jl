using Flux
using Flux: train!, Dense, Descent
using Statistics
using Statistics: mean
using Plots

target_function(x) = π * x + ℯ

x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = target_function.(x_train), target_function.(x_test)

model = Dense(1 => 1)
predict = Dense(1 => 1)

loss(model, x, y) = mean(abs2.(model(x) .- y))
opt = Descent()

@info "Initial Loss: $(loss(predict, x_test, y_test))"
data = [(x_train, y_train)]
NUM_EPOCHS = 200
for epoch in 1:NUM_EPOCHS
    train!(loss, predict, data, opt)
    if epoch % NUM_EPOCHS//10 == 0
        @info "Epoch $(epoch) --> $(loss(predict, x_test, y_test))"
    end
end

x = 1:200
y = transpose(predict(hcat(x...)))
plt = plot(1:200, y)
savefig(plt, "./my_plot.png")
display(plt)
@info "Done"