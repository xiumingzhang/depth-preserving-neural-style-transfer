require 'torch'
require 'nn'
local layer_utils = require 'fast_neural_style.layer_utils'
local crit, parent = torch.class('nn.DepthCriterion', 'nn.Criterion')


--[[
Input: args is a table with the following keys:
- cnn: A network giving the base CNN.
- content_layers: An array of layer strings
- content_weights: A list of the same length as content_layers
- style_layers: An array of layers strings
- style_weights: A list of the same length as style_layers
- agg_type: What type of spatial aggregaton to use for style loss;
  "mean" or "gram"
- deepdream_layers: Array of layer strings
- deepdream_weights: List of the same length as deepdream_layers
- loss_type: Either "L2", or "SmoothL1"
--]]
function crit:__init(args)
  
  self.net = args.cnn
  self.net:evaluate()
  
  layer_utils.trim_network(self.net)
  self.grad_net_output = torch.Tensor()
end

--[[
target: Tensor of shape (1, 3, H, W) giving pixels for style target image
--]]
function crit:setDepthTarget(target)
  self.net:forward(target)
end


--[[
target: Tensor of shape (N, 3, H, W) giving pixels for content target images
--]]
function crit:setContentTarget(target)
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('none')
  end
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('capture')
  end
  self.net:forward(target)
end


function crit:setStyleWeight(weight)
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer.strength = weight
  end
end


function crit:setContentWeight(weight)
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer.strength = weight
  end
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
    self:setContentTarget(target.content_target)
  end
  if target.style_target then
    self.setStyleTarget(target.style_target)
  end
  -- TODO: do we need to set depth target here?


  -- Make sure to set all content and style loss layers to loss mode before
  -- running the image forward.
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('loss')
  end
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('loss')
  end
  --[[
  for i, depth_loss_layer in ipairs(self.depth_loss_layers) do
    depth_loss_layer:setMode('loss')
  end --]]

  local output = self.net:forward(input)

  -- Set up a tensor of zeros to pass as gradient to net in backward pass
  self.grad_net_output:resizeAs(output):zero()

  -- Go through and add up losses
  self.total_content_loss = 0
  self.content_losses = {}
  self.total_style_loss = 0
  self.style_losses = {}
  --[[self.total_depth_loss = 0
  self.depth_losses = {} --]]
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    self.total_content_loss = self.total_content_loss + content_loss_layer.loss
    table.insert(self.content_losses, content_loss_layer.loss)
  end
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    self.total_style_loss = self.total_style_loss + style_loss_layer.loss
    table.insert(self.style_losses, style_loss_layer.loss)
  end
  --[[
  for i, depth_loss_layer in ipairs(self.depth_loss_layers) do
    self.total_depth_loss = self.total_depth_loss + depth_loss_layer.loss
    table.insert(self.depth_losses, depth_loss_layer.loss)
  end--]]
  
  self.output = self.total_style_loss + self.total_content_loss --+ self.total_depth_loss  -- we need to modify this
  return self.output
end


function crit:updateGradInput(input, target)
  self.gradInput = self.net:updateGradInput(input, self.grad_net_output)
  return self.gradInput
end

