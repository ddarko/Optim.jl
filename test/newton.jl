using Optim
using Base.Test

function f_1(x::Vector)
    (x[1] - 5.0)^4
end

function g!_1(x::Vector, storage::Vector)
    storage[1] = 4.0 * (x[1] - 5.0)^3
end

function h!_1(x::Vector, storage::Matrix)
    storage[1, 1] = 12.0 * (x[1] - 5.0)^2
end

d = TwiceDifferentiableFunction(f_1, g!_1, h!_1)

@test_throws ArgumentError Optim.optimize(f_1, [0.0], method=Newton())
@test_throws ArgumentError Optim.optimize(DifferentiableFunction(f_1, g!_1), [0.0], method=Newton())

results = Optim.optimize(d, [0.0], method=Newton())
@assert length(results.trace.states) == 0
@assert results.gr_converged
@assert norm(results.minimum - [5.0]) < 0.01

eta = 0.9

function f_2(x::Vector)
  (1.0 / 2.0) * (x[1]^2 + eta * x[2]^2)
end

function g!_2(x::Vector, storage::Vector)
  storage[1] = x[1]
  storage[2] = eta * x[2]
end

function h!_2(x::Vector, storage::Matrix)
  storage[1, 1] = 1.0
  storage[1, 2] = 0.0
  storage[2, 1] = 0.0
  storage[2, 2] = eta
end

d = TwiceDifferentiableFunction(f_2, g!_2, h!_2)
results = Optim.optimize(d, [127.0, 921.0], method=Newton())
@assert length(results.trace.states) == 0
@assert results.gr_converged
@assert norm(results.minimum - [0.0, 0.0]) < 0.01

# Test Optim.newton for all twice differentiable functions in Optim.UnconstrainedProblems.examples
for (name, prob) in Optim.UnconstrainedProblems.examples
	if prob.istwicedifferentiable
		ddf = TwiceDifferentiableFunction(prob.f, prob.g!,prob.h!)
		res = Optim.optimize(ddf, prob.initial_x, method=Newton())
		@assert norm(res.minimum - prob.solutions) < 1e-2
	end
end

using ForwardDiff
easom(x) = -cos(x[1])*cos(x[2])*exp(-((x[1]-pi)^2 + (x[2]-pi)^2))
x, y = 1.2:0.1:4, 1.2:0.1:4

g_easom = ForwardDiff.gradient(easom)
h_easom = ForwardDiff.hessian(easom)

# start where old Newton's method would fail due to concavity 
optimize(easom, (x, y) -> copy!(y, g_easom(x)), (x,y)->copy!(y, h_easom(x)), [2., 2.], Newton())
@test_approx_eq optimize(easom, (x, y) -> copy!(y, g_easom(x)), (x,y)->copy!(y, h_easom(x)), [2., 2.], Newton()).minimum [float(pi);float(pi)]
