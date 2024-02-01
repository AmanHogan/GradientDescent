from gradient_descent import (GradientDescent, StochasticGradient, BatchGradient, cmd_parser)
from gradient_grapher import (plot_gradient, plot_loss)

args = cmd_parser()
gradient_object = None

if args.gtype == 's':
    gradient_object = StochasticGradient(args.filename, args.samples, args.features, args.columns, args.epoch, args.alpha, args.gtype)
    gradient_object.sgd()
else:
    gradient_object = BatchGradient(args.filename, args.samples, args.features, args.columns, args.epoch, args.alpha, args.gtype)
    gradient_object.bgd()



print(f"Final Parameters: {gradient_object.w}")
print(f"Starting Error {gradient_object.errors[0]}")
print(f"Final Error {gradient_object.errors[len(gradient_object.errors)-1]}")
print(f"Smallest Error {gradient_object.errors[len(gradient_object.errors)-1]}")
print(f"# of Epochs: {gradient_object.epochs}")
print(f"Gradient Performed {gradient_object.gtype}")
print(f"Best W (Params) {gradient_object.best_params} at Epoch {gradient_object.errors.index(min(gradient_object.errors))}")


plot_loss(gradient_object)
plot_gradient(gradient_object)
