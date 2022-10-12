#using Core: Vector
using Random
	
#=
Multiline comment.
=#

"""Generate AND data with input and output vectors."""
function generate_and_data(n::Int)::Tuple{Matrix, Vector}
	x::Matrix = rand([0, 1], n, 2)
	y = Vector{Int}(undef, size(x, 1))
	
	for i = 1:size(x, 1)
		if x[i, :] == ones(Int, 2)
			y[i] = 1
		else
			y[i] = 0
		end
	end
	
	return x, y
end
	

"""Generate XOR data."""
function generate_xor_data(n::Int)::Tuple{Matrix, Vector}
	x::Matrix = rand([0, 1], n, 2)
	y = Vector{Int}(undef, size(x, 1))
	
	for i = 1:size(x, 1)
		if x[i, :] == [1, 0] ||
		   x[i, :] == [0, 1]
			y[i] = 1
		else
			y[i] = 0
		end
	end
	
	return x, y
end

# use \euler for Unicode e or Base.MathConstants.e
# or use exp for ℯ ^ ()
sigmoid(x::Real)::Real = 1 / (1 + exp(-x))

mutable struct Node
	activation_func::Function
	# IDEA: Matrix represetation for efficiency
	outgoing_nodes::Vector{Node}
	outgoing_weights::Vector{Float16}
	# IDEA: keeping track of incoming nodes may be too expensive
	incoming_nodes::Vector{Node}
	in_weights::Vector{Float16}
	# IDEA: distance to nodes
	# IDEA: angle/direction nodes
	# IDEA: higher-dimensional node connections
end

mutable struct TrackFiring
	# keep track of n*m nodes
	# then report back a visualization if a threshold of activations
	# or a frequency threshold were surpased
	# enable visualization of particularly interesting firing patters
end

mutable struct TrackGrowth
	# keep track of n*m nodes
	# report back if a neuron generation
	# or degeneration pattern emerged
end

mutable struct Model
	layers::Vector{Union{Nothing, HiddenLayer, InputLayer, OutputLayer}}
end

# IDEA: option for randomness
# IDEA: option for distribution to draw randomness from
# IDEA: option to create different shapes of initial networks
# IDEA: optin to scale the initial graph size
function init_square_model(width::Int, depth::Int)::Model
	# during build time the worm traverses the network back to front
	grid = Matrix{Node, (width, depth)}


	# NOTE: Layers don't make sense for architecture of a brain
	# we see layers emerge after the fact, they are emergent
	# find good graph initialization algorithm
	# give nodes positions in n-dimensional space 
		for x in 1:width
		old_node = nothing

		for y in 1:depth
			out_weight = rand(Float64, 1)
			node = Node(sigmoid, 
				Vector{old_node},
				Vector{in_weight},
				nothing,
				nothing)
			old_node = node
			grid[x, y] = node
		end
	end

	# forward linking
	# sideways linking

	n1 = InputNode(sigmoid, 1, nothing, 0.5)
	n3 = OutputNode(sigmoid, n2, 0.5, 1)
	n1.out_node = n2
	n2.out_node = n3
	return n1, n2, n3
end

function forward_pass()::Model

end

function backward_pass() end

function train_model(iterations::Int)::Model
	for i in iterations
		forward_pass()
		backward_pass()
	end
end

function test_model() end	
	

x_train, y_train = generate_xor_data(5)
println(x_train, y_train)
n1, n2, n3 = init_square_model(10, 10)

