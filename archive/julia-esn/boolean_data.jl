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