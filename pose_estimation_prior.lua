

require 'paths'
require 'nngraph'
require 'cunn'
require 'cudnn'
local ffi = require 'ffi'

paths.dofile('util.lua')
paths.dofile('img.lua')


m = torch.load('umich-stacked-hourglass.t7')
list_file = '/nfs.yoda/xiaolonw/torch_projects/my-eyescream/affordance3/pose_estimation/makelist/friends_matches_testposelist2.txt'
prior_images = '/nfs.yoda/xiaolonw/dcgan/results_afford_scale_pred_priors/'
path_dataset = '/scratch/xiaolonw/affordance_general/data2/'
savepath = './samples2/'


local testnum = 2834
local maxPathLength = 300
local loadSize = 256

lblset = torch.Tensor(testnum, 34):float():fill(0)
offsets = torch.Tensor(testnum, 2):float():fill(0)
imagePath = torch.CharTensor()
imagePath:resize(testnum, maxPathLength):fill(0)

local s_data = imagePath:data()


f = assert(io.open(list_file, "r"))
for i = 1, testnum do 

  -- get name
  list = f:read("*line")
  cnt = 0 
  for str in string.gmatch(list, "%S+") do
    -- print(str)
    cnt = cnt + 1
    if cnt == 1 then 
      filename = str
    elseif cnt <= 35 then 
      lblset[{{i}, {cnt - 1} }] = tonumber(str)
    else
      offsets[{{i},{cnt - 35}}] = tonumber(str)
    end

  end
  assert(cnt == 37)

  filename = path_dataset .. filename 
  ffi.copy(s_data, filename)
  s_data = s_data + maxPathLength


  if i % 1000 == 0 then
    print(i)
    print(ffi.string(torch.data(imagePath[i])))
    -- print(ffi.string(torch.data(self.labelPath[i])) )

  end

end

f:close()


for i = 1, testnum do 

	print(i)

	local imname = ffi.string(torch.data(imagePath[i]))
	local im = image.load(imname)

	local x_array = torch.Tensor(17):float()
	local y_array = torch.Tensor(17):float()
	local now_array = lblset[i]

	local height = im:size()[2]
	local width  = im:size()[3]

	for j = 1, 17 do 
		x_array[j] = now_array[j * 2 - 1]
		y_array[j] = now_array[j * 2 ]
	end

  masks = torch.Tensor(16, height, width)

  for j = 1, 16 do 

    imname_mask = string.format('%04d_%04d_hm.jpg', i - 1, j )
    imname_mask = prior_images .. imname_mask 
    mask = image.load(imname_mask)
    masks[{{j}, {}, {}}]:copy(mask) 

  end

	local minx = math.max(x_array:min() - 50, 1)
	local miny = math.max(y_array:min() - 50, 1)
	local maxx = math.min(x_array:max() + 50, width)
	local maxy = math.min(y_array:max() + 50, height)

  minx = math.floor(minx)
  miny = math.floor(miny)
  maxx = math.floor(maxx)
  maxy = math.floor(maxy)

	local pose_height = ( maxy - miny )
	local pose_width  = ( maxx - minx )

	local pad = math.max( (pose_height - pose_width) / 2, 0 )
	minx = math.max(minx - pad, 1)
	maxx = math.min(maxx + pad, width)
  pose_width  = ( maxx - minx )

  local minx2 = minx - math.floor(offsets[i][1])
  local miny2 = miny - math.floor(offsets[i][2])
  local maxx2 = maxx - math.floor(offsets[i][1])
  local maxy2 = maxy - math.floor(offsets[i][2])

  local offx = 1
  local offy = 1
  local dimx = pose_width + 1
  local dimy = pose_height + 1

  if minx2 < 0 then 
    offx = - minx2 + 1
    minx2 = 1
  end
  
  if miny2 < 0 then 
    offy = - miny2 + 1
    miny2 = 1
  end

  if maxx2 > width then 
    dimx = dimx - (maxx2 - width)
    maxx2 = width
  end

  if maxy2 > height then 
    dimy = dimy - (maxy2 - height) 
    maxy2 = height
  end

  masks_pose = torch.Tensor(16, pose_height + 1, pose_width + 1)
  

  if dimx - offx ~= maxx2 - minx2 then 
    dimx = maxx2 - minx2 + offx
  end

  if dimy - offy ~= maxy2 - miny2 then 
    dimy = offy + maxy2 - miny2
  end



  masks_pose[{{}, {offy, dimy}, {offx, dimx}}]:copy( masks[{{}, {miny2, maxy2}, {minx2, maxx2}}] ) 



  masks_pose2 = torch.Tensor(16, 64, 64):float():fill(0)
  for i = 1 , 16 do
    masks_pose2[{{i}, {}, {}}]:copy(image.scale(masks_pose[i], 64, 64))
  end


	local ratioh = pose_height / 256.0
	local ratiow = pose_width  / 256.0

 
	local crop_img = im[{{}, {miny, maxy}, {minx, maxx}}] 
	local inp = image.scale(crop_img, loadSize, loadSize)

	local out = m:forward(inp:view(1,3,256,256):cuda())

  cutorch.synchronize()
  local hm = out[2][1]:float()
  hm[hm:lt(0)] = 0

  hm = hm + masks_pose2 * 0.25



  local fake_center = torch.Tensor(2)
  fake_center[1] = 300
  fake_center[2] = 300

  local preds_hm, preds_img = getPreds(hm, fake_center, 1.0)

  preds_hm:mul(4)
  preds_hm[{{}, {}, {1}}] = preds_hm[{{}, {}, {1}}] * ratiow + minx
  preds_hm[{{}, {}, {2}}] = preds_hm[{{}, {}, {2}}] * ratioh + miny



  local dispImg = drawOutput(im, hm, preds_hm[1])
  local imgname = paths.concat(savepath, string.format('%04d.jpg', i ))

  image.save(imgname, dispImg )


  collectgarbage()







end























