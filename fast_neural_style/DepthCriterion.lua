require 'torch'
require 'nn'

require 'fast_neural_style.DepthLoss'

local layer_utils = require 'fast_neural_style.layer_utils'

local crit, parent = torch.class('nn.DepthCriterion', 'nn.Criterion')


--[[
Input: args is a table with the following keys:
- cnn: A network giving the base CNN.
- content_layers: An array of layer strings
- content_weights: A list of the same length as content_layers
- style_layers: An array of layers strings
- style_weights: A list of the same length as style_layers
- deepdream_layers: Array of layer strings
- deepdream_weights: List of the same length as deepdream_layers
- loss_type: Either "L2", or "SmoothL1"
--]]
function crit:__init(args)
  
  args.depth_layers = args.depth_layers or {}
  
  self.net = args.cnn
  self.net:evaluate()
  self.depth_loss_layers = {}
  
  for i, layer_string in ipairs(args.depth_layers) do
    local weight = args.depth_weights[i]
    local depth_loss_layer = nn.DepthLoss(weight, args.loss_type)
    layer_utils.insert_after(self.net, layer_string, depth_loss_layer)
    table.insert(self.depth_loss_layers, depth_loss_layer)
  end
  
  -- layer_utils.trim_network(self.net)
  self.grad_net_output = torch.Tensor()
end

--[[
target: Tensor of shape (1, 3, H, W) giving pixels for style target image
--]]
function crit:setDepthTarget(target)
  self.net:forward(target)
end



--[[
Inputs:
- input: Tensor of shape (N, 3, H, W) giving pixels for generated images
- target: Table with the following keys:
  - content_target: Tensor of shape (N, 3, H, W)
  - style_target: Tensor of shape (1, 3, H, W)
--]]
function crit:updateOutput(input, target)
  if target.content_target then
    self:setDepthTarget(target.content_target)
  end
  
  local output = self.net:forward(input)
  
  -- Compute self.loss
  self.loss = self.crit:forward(output, self.target_output)

  -- Set up a tensor of zeros to pass as gradient to net in backward pass
  self.grad_net_output:resizeAs(output):zero()
  
  self.total_depth_loss = 0
  self.depth_losses = {}
  
  for i, depth_loss_layer in ipairs(self.depth_loss_layers) do
    self.total_depth_loss = self.total_depth_loss + depth_loss_layer.loss
    table.insert(self.depth_losses, depth_loss_layer.loss)
  end
  
  self.output = self.total_depth_loss
  return self.output
end


function crit:updateGradInput(input, target)
  self.gradInput = self.net:updateGradInput(input, self.grad_net_output)
  return self.gradInput
end