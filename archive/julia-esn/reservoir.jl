include("constants.jl")
include("boolean_data.jl")

using StatsBase
using DataFrames
using Distributions

mutable struct Node
	activation_func::Function
    activation::Float64
end

function get_out_nodes(in_nodes)
    sample_source = setdiff(1:NUM_NODES, in_nodes)
    out_nodes = sample(sample_source, NUM_OUT_NODES, replace = false)
    return out_nodes
end

function init_nodes()
    nodes = Array{Node}(undef, NUM_NODES)

    # sigmoid activation function
    σ(x) = 1 / (1 + exp(-x))
    # Google's swish activation function
    swish(x) = x * σ(x)

    for i in 1:NUM_NODES
        nodes[i] = Node(swish, 0)
    end

    return nodes
end

# forward a single tick
function forward(nodes, adj_matrix, in_nodes, out_nodes)

    println("PRE ACTIVATION MAP")
    print_activation_map(nodes)

    # update the activation of non input nodes
    for i in filter(node -> node ∉ in_nodes, 1:NUM_NODES)
        # find nodes pointing to current node
        non_zero_indices = findall(x -> x > 0, adj_matrix[:,i])
        for j in non_zero_indices
            # from node j to node i
            w_to_i = adj_matrix[j,i]
            act_j = nodes[j].activation
            nodes[i].activation += w_to_i * act_j
            # TODO: apply activation function
        end
    end

    println("POST ACTIVATION MAP")
    print_activation_map(nodes)

    # return the activation of the output nodes
    output_nodes = nodes[1:end .∈ out_nodes]
    activation = [x.activation for x in output_nodes]
    return activation
end

function print_activation_map(nodes)
    activation_list = map(x -> x.activation, nodes)
    println("$activation_list\n")
end

# feed signal into reservoir 
function apply_signal(x, y, nodes, adj_matrix, in_nodes, out_nodes)
    # initialize empty dataframe to evaluate output
    df = DataFrame(target=Int[], label=Float64[], error=Float64[])

    # input the signal tick by tick
    for (input, target) in zip(eachrow(x), y)
        # apply input signal to the input nodes at current tick
        println(input, " ", target)
        for (key, val) in zip(in_nodes, input)
            nodes[key].activation = val
            println(key, " ", val)
            # TODO: apply activation function
        end
        
        label = forward(nodes, adj_matrix, in_nodes, out_nodes)
        error = target .- label

        # update dataframe
        # TODO: generalize label[1] and error[1] to n dimensions
        println("target: $target, label: $label, error: $error")
        push!(df, [target, label[1], error[1]])
    end

    return df
end

# TODO: graph theory stats

function main()
    # TODO: struct model
    
    # initialize truncated normal distribution
    d = truncated(Normal(NORMAL_MU, NORMAL_SIGMA), MIN_NODE_WEIGHT, MAX_NODE_WEIGHT)
    # generate weighted adjacency matrix
    adj_matrix = rand(d, NUM_NODES, NUM_NODES)
    
    # choose unique input nodes of an adjacency matrix
    in_nodes = sample(1:NUM_NODES, NUM_IN_NODES, replace = false)
    # choose unique output nodes of adjacency matrix
    out_nodes = get_out_nodes(in_nodes)
    # initialize nodes
    nodes = init_nodes()
    # initialize signal and target
    #x, y = generate_and_data(SIGNAL_LENGTH)
    x, y = ([1 1; 1 1; 0 0; 0 0; 1 1; 1 1; 0 0], [1, 1, 0, 0, 1, 1, 0])
    # pass signal through reservoir
    df = apply_signal(x, y, nodes, adj_matrix, in_nodes, out_nodes)

    # linear fit

    println("Adjacency matrix has dim: $NUM_NODES")
    println("Number of input nodes: $NUM_IN_NODES")
    println("Number of output nodes: $NUM_OUT_NODES")
    println("Max node weight: $MAX_NODE_WEIGHT")
    println("Signal length: $SIGNAL_LENGTH")

    println(df)

end

main()