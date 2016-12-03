require 'torch'
require 'nn'

require 'fast_neural_style.GramMatrix'
require 'path'

paths.dofile('../relative-depth/models/hourglass3.lua')
-- Depth Loss NN goes here

function DepthLoss:__init(strength, loss_type, agg_type)
    self.g_model = get_model()
    self.g_model.period = 1
    self.g_model = g_model:cuda()

end

function DepthLoss:updateOutput(input)
  -- todo: give train_loader inputs
  network_input_height = 240
  network_input_width = 320
  _batch_input_cpu = torch.Tensor(1,3,network_input_height,network_input_width)


  n_thresh = 140;
  thresh = torch.Tensor(n_thresh);
  for i = 1, n_thresh do
      thresh[i] = 0.1 + i * 0.01;
  end
  local img = input -- make sure input is an image
  local img_original_height = img:size(2)
  local img_original_width = img:size(3)
  _batch_input_cpu[{1,{}}]:copy( crop_resize_input(img) )
  g_model:evaluate() -- from test_model_on_NYU.lua
  local _single_data = {};
  _single_data[1] = data_handle[i]
  -- forward
  local batch_output = model:forward(_batch_input_cpu:cuda());  
  cutorch.synchronize()
  local temp = batch_output
  if torch.type(batch_output) == 'table' then
        batch_output = batch_output[1]
  end
  local original_size_output = torch.Tensor(1,1,img_original_height, img_original_width)
  --image.scale(src, width, height, [mode])    Scale it to the original size!
  original_size_output[{1,1,{}}]:copy( inpaint_pad_output_our(batch_output, img_original_width, img_original_height) ) 
        
        -- evaluate on the original size!
        _evaluate_correctness_our(original_size_output, _single_data[1], WKDR[{i,{}}], WKDR_eq[{i,{}}], WKDR_neq[{i,{}}]);


        local gtz_h5_handle = hdf5.open(paths.dirname(data_handle[i].img_filename) .. '/' .. i ..'_depth.h5', 'r')
        local gtz = gtz_h5_handle:read('/depth'):all()
        gtz_h5_handle:close()
        assert(gtz:size(1) == 480)
        assert(gtz:size(2) == 640)
        
        -- transform the output depth with training mean and std
        transformed_weifeng_z_orig_size = normalize_output_depth_with_NYU_mean_std( original_size_output[{1,1,{}}] )

        -- evaluate the data at the cropped area
        local metric_test_crop = 16
        transformed_weifeng_z_orig_size = transformed_weifeng_z_orig_size:sub(metric_test_crop,img_original_height-metric_test_crop,metric_test_crop,img_original_width-metric_test_crop)
        gtz = gtz:sub(metric_test_crop,img_original_height-metric_test_crop,metric_test_crop,img_original_width-metric_test_crop)
        
        -- metric error
        fmse[i], fmselog[i], flsi[i], fabsrel[i], fsqrrel[i] = metric_error(gtz, transformed_weifeng_z_orig_size)
      local local_image = torch.Tensor(1,img_original_height,img_original_width)
        local local_image2 = torch.Tensor(3,img_original_height,img_original_width)
        local output_image = torch.Tensor(3, img_original_height,img_original_width * 2)


        local_image:copy(original_size_output:double())
        local_image = local_image:add( - torch.min(local_image) )
        local_image = local_image:div( torch.max(local_image:sub(1,-1, 20, img_original_height - 20, 20, img_original_width - 20)) )
        

        output_image[{1,{1,img_original_height},{img_original_width + 1,img_original_width * 2}}]:copy(local_image)
        output_image[{2,{1,img_original_height},{img_original_width + 1,img_original_width * 2}}]:copy(local_image)
        output_image[{3,{1,img_original_height},{img_original_width + 1,img_original_width * 2}}]:copy(local_image)


        local_image2:copy(image.load(data_handle[i].img_filename)) 

        output_image[{{1},{1,img_original_height},{1,img_original_width}}]:copy(local_image2[{1,{}}])
        output_image[{{2},{1,img_original_height},{1,img_original_width}}]:copy(local_image2[{2,{}}])
        output_image[{{3},{1,img_original_height},{1,img_original_width}}]:copy(local_image2[{3,{}}])
        -- output_image -- not sure if we need to output an image
        -- TODO: then we need to compute the per-pixel distance.
        -- and return this distance
end

function DepthLoss:updateGradInput(input, gradOutput)

end

function DepthLoss:setMode(mode)
  if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end
