require 'torch'
require 'nn'

require 'fast_neural_style.GramMatrix'
-- Depth Loss NN goes here

function DepthLoss:__init(strength, loss_type, agg_type)

end

function DepthLoss:updateOutput(input)

end

function DepthLoss:updateGradInput(input, gradOutput)
end

function DepthLoss:setMode(mode)
  if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end
